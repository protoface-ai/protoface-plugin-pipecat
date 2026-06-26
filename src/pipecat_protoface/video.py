"""Pipecat video service for Protoface avatars."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import aiohttp
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings

from ._client import (
    PROTOFACE_INPUT_SAMPLE_RATE,
    ProtofaceAudioFrame,
    ProtofaceException,
    ProtofaceMediaClient,
    ProtofaceMediaFrame,
    ProtofaceRelayClient,
    ProtofaceVideoFrame,
)

_BYTES_PER_SAMPLE = 2
_DEFAULT_AUDIO_CHUNK_MS = 40
_DEFAULT_AUDIO_SEND_AHEAD_MS = 1000
_DEFAULT_CLIENT_READY_TIMEOUT_SECS = 30.0
_STOP_CLIENT_READY_GRACE_SECS = 0.25
_MAX_PENDING_AUDIO_BYTES = 8 * 1024 * 1024
_MAX_PENDING_AUDIO_EVENTS = 256
_MAX_PENDING_MEDIA_FRAMES = 64


def _debug_media(message: str) -> None:
    if os.environ.get("PROTOFACE_DEBUG_MEDIA"):
        print(f"[pipecat-protoface] {message}", flush=True)


@dataclass
class ProtofaceVideoSettings(ServiceSettings):
    """Runtime settings for the Protoface Pipecat video service."""

    audio_chunk_ms: int = _DEFAULT_AUDIO_CHUNK_MS
    audio_send_ahead_ms: int = _DEFAULT_AUDIO_SEND_AHEAD_MS
    client_ready_timeout_secs: float = _DEFAULT_CLIENT_READY_TIMEOUT_SECS


@dataclass(slots=True)
class _AudioChunk:
    audio: bytes
    sample_rate: int
    num_channels: int


@dataclass(slots=True)
class _FlushAudio:
    pass


_AudioQueueItem = _AudioChunk | _FlushAudio
_PendingAudioEvent = TTSAudioRawFrame | _FlushAudio
_PendingMediaFrame = ProtofaceMediaFrame


class ProtofaceVideoService(AIService):
    """Pipecat video service for hosted Protoface avatars.

    TTS audio enters as ``TTSAudioRawFrame``. Synchronized avatar image and
    speech audio frames are emitted downstream.
    """

    Settings = ProtofaceVideoSettings

    def __init__(
        self,
        *,
        api_key: str,
        avatar_id: str,
        api_url: str | None = None,
        session: aiohttp.ClientSession | None = None,
        media_client: ProtofaceMediaClient | None = None,
        max_duration_seconds: int | None = None,
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
        settings: ProtofaceVideoSettings | None = None,
        **kwargs: Any,
    ) -> None:
        default_settings = ProtofaceVideoSettings(model=None)
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(settings=default_settings, **kwargs)

        self._avatar_id = avatar_id
        self._max_duration_seconds = max_duration_seconds
        self._metadata = dict(metadata or {})
        self._client = media_client or ProtofaceRelayClient(
            api_key=api_key,
            api_url=api_url,
            session=session,
        )
        self._resampler = create_stream_resampler()
        self._audio_buffer = bytearray()
        self._audio_buffer_sample_rate = PROTOFACE_INPUT_SAMPLE_RATE
        self._audio_buffer_channels = 1
        self._queue: asyncio.Queue[_AudioQueueItem] = asyncio.Queue()
        self._connect_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._media_task: asyncio.Task[None] | None = None
        self._transport_ready = False
        self._client_ready_event = asyncio.Event()
        self._fatal_error: Exception | None = None
        self._fatal_error_reported = False
        self._pending_audio_events: list[_PendingAudioEvent] = []
        self._pending_audio_bytes = 0
        self._pending_media_frames: deque[_PendingMediaFrame] = deque()
        self._audio_state_lock = asyncio.Lock()
        self._media_state_lock = asyncio.Lock()
        self._flushing_pending_media = False
        self._media_generation = 0
        self._next_audio_send_at = 0.0
        self._should_measure_ttfb = False
        self._ttfb_metrics_active = False
        self._sent_audio_chunks = 0
        self._pushed_audio_frames = 0
        self._pushed_video_frames = 0

    def can_generate_metrics(self) -> bool:
        """Protoface can report TTFB through the Pipecat service hooks."""

        return True

    async def start(self, frame: StartFrame) -> None:
        """Start the hosted Protoface avatar session."""

        if self._has_runtime_state():
            await self._teardown_runtime(cancel_client=True)
        await super().start(frame)
        self._client_ready_event.clear()
        self._transport_ready = False
        self._fatal_error = None
        self._fatal_error_reported = False
        self._flushing_pending_media = False
        self._media_generation += 1
        self._should_measure_ttfb = False
        self._ttfb_metrics_active = False
        self._resampler = create_stream_resampler()
        await self._create_send_task()
        self._connect_task = self.create_task(self._connect_client())

    async def stop(self, frame: EndFrame) -> None:
        """Stop the hosted Protoface avatar session."""

        await super().stop(frame)
        if (
            not self._client_ready_event.is_set()
            and self._connect_task is not None
            and not self._connect_task.done()
        ):
            timeout = min(
                cast(ProtofaceVideoSettings, self._settings).client_ready_timeout_secs,
                _STOP_CLIENT_READY_GRACE_SECS,
            )
            await self._wait_for_client_ready(timeout_secs=timeout)
        if self._client_ready_event.is_set():
            await self._flush_audio(report_ready_timeout=False)
        await self._cancel_connect_task()
        await self._cancel_task_attr("_media_task")
        await self._client.stop()
        await self._teardown_runtime(cancel_client=False)

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the hosted Protoface avatar session."""

        await super().cancel(frame)
        if self._has_runtime_state():
            await self._interrupt_avatar_output(restart_send_task=False)
        await self._teardown_runtime(cancel_client=True)

    def _has_runtime_state(self) -> bool:
        return (
            self._connect_task is not None
            or self._send_task is not None
            or self._media_task is not None
            or self._client_ready_event.is_set()
            or self._transport_ready
            or bool(self._audio_buffer)
            or not self._queue.empty()
            or bool(self._pending_audio_events)
            or bool(self._pending_media_frames)
        )

    async def _teardown_runtime(self, *, cancel_client: bool) -> None:
        await self._cancel_connect_task()
        await self._cancel_tasks()
        if cancel_client:
            await self._client.cancel()
        self._audio_buffer.clear()
        await self._drain_audio_queue()
        self._pending_audio_events.clear()
        self._pending_audio_bytes = 0
        self._pending_media_frames.clear()
        self._client_ready_event.clear()
        self._transport_ready = False
        self._fatal_error = None
        self._fatal_error_reported = False
        self._flushing_pending_media = False
        self._media_generation += 1
        self._next_audio_send_at = 0.0
        self._should_measure_ttfb = False
        self._ttfb_metrics_active = False
        self._resampler = create_stream_resampler()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process Pipecat frames through the Protoface avatar service."""

        await super().process_frame(frame, direction)
        if isinstance(frame, OutputTransportReadyFrame):
            await self.push_frame(frame, direction)
            async with self._media_state_lock:
                self._transport_ready = True
                generation = self._media_generation
                self._flushing_pending_media = bool(self._pending_media_frames)
            await self._flush_pending_media(generation)
        elif isinstance(frame, TTSStartedFrame):
            self._should_measure_ttfb = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._handle_audio_frame(frame)
        elif isinstance(frame, TTSStoppedFrame):
            await self._flush_audio()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame | UserStartedSpeakingFrame):
            await self._handle_interruption()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _connect_client(self) -> None:
        try:
            await self._client.start(
                avatar_id=self._avatar_id,
                max_duration_seconds=self._max_duration_seconds,
                metadata=self._metadata,
            )
            await self._create_consume_tasks()
            await self._process_pending_audio_events()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_fatal("Protoface avatar session failed to start", exc)

    async def _create_consume_tasks(self) -> None:
        if self._media_task is None or self._media_task.done():
            self._media_task = self.create_task(self._consume_media())

    async def _cancel_tasks(self) -> None:
        for attr in ("_send_task", "_media_task"):
            await self._cancel_task_attr(attr)

    async def _cancel_connect_task(self) -> None:
        await self._cancel_task_attr("_connect_task")

    async def _cancel_task_attr(self, attr: str) -> None:
        task = getattr(self, attr)
        if task is None:
            return
        if task.done():
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                pass
        else:
            await self.cancel_task(task)
        setattr(self, attr, None)

    async def _handle_audio_frame(self, frame: TTSAudioRawFrame) -> None:
        overflow: ProtofaceException | None = None
        async with self._audio_state_lock:
            if self._fatal_error is not None:
                return
            if not self._client_ready_event.is_set():
                overflow = self._append_pending_audio_event_locked(frame)
            else:
                await self._enqueue_audio_frame_locked(frame)
        if overflow is not None:
            await self._fail_fatal("Protoface avatar pending audio buffer overflow", overflow)

    async def _enqueue_audio_frame_locked(
        self,
        frame: TTSAudioRawFrame,
        *,
        drop_if_not_ready: bool = True,
    ) -> None:
        target_sample_rate = self._client.input_sample_rate or PROTOFACE_INPUT_SAMPLE_RATE
        if self._audio_buffer and (
            target_sample_rate != self._audio_buffer_sample_rate
            or frame.num_channels != self._audio_buffer_channels
        ):
            await self._flush_audio_locked(
                wait_for_ready=False,
                enqueue_flush=True,
                drop_if_not_ready=drop_if_not_ready,
            )
        self._audio_buffer_sample_rate = target_sample_rate
        self._audio_buffer_channels = frame.num_channels

        audio = await self._resampler.resample(frame.audio, frame.sample_rate, target_sample_rate)
        self._audio_buffer.extend(audio)

        chunk_size = int(
            target_sample_rate
            * max(1, cast(ProtofaceVideoSettings, self._settings).audio_chunk_ms)
            / 1000
            * _BYTES_PER_SAMPLE
            * frame.num_channels
        )
        while len(self._audio_buffer) >= chunk_size:
            chunk = bytes(self._audio_buffer[:chunk_size])
            del self._audio_buffer[:chunk_size]
            await self._queue.put(
                _AudioChunk(
                    audio=chunk,
                    sample_rate=target_sample_rate,
                    num_channels=frame.num_channels,
                )
            )

    async def _flush_audio(
        self,
        *,
        wait_for_ready: bool = True,
        report_ready_timeout: bool = True,
    ) -> None:
        wait_for_client_ready = False
        wait_for_queue = False
        overflow: ProtofaceException | None = None
        async with self._audio_state_lock:
            wait_for_client_ready, wait_for_queue, overflow = await self._flush_audio_locked(
                wait_for_ready=wait_for_ready,
                enqueue_flush=True,
            )
        if overflow is not None:
            await self._fail_fatal("Protoface avatar pending audio buffer overflow", overflow)
            return
        if wait_for_client_ready:
            if not await self._wait_for_client_ready():
                if report_ready_timeout:
                    exc = TimeoutError("Timed out waiting for Protoface avatar session to start.")
                    await self._cancel_connect_task()
                    await self._fail_fatal("Protoface avatar session failed to start", exc)
                return
            if self._fatal_error is not None:
                return
            wait_for_queue = True
        if wait_for_queue:
            await self._queue.join()

    async def _wait_for_client_ready(self, *, timeout_secs: float | None = None) -> bool:
        timeout = max(
            0.0,
            (
                cast(ProtofaceVideoSettings, self._settings).client_ready_timeout_secs
                if timeout_secs is None
                else timeout_secs
            ),
        )
        if timeout <= 0:
            return self._client_ready_event.is_set()
        try:
            await asyncio.wait_for(self._client_ready_event.wait(), timeout=timeout)
        except TimeoutError:
            return self._client_ready_event.is_set()
        return True

    async def _flush_audio_locked(
        self,
        *,
        wait_for_ready: bool = True,
        enqueue_flush: bool = True,
        drop_if_not_ready: bool = True,
    ) -> tuple[bool, bool, ProtofaceException | None]:
        if self._fatal_error is not None:
            self._audio_buffer.clear()
            await self._drain_audio_queue()
            return False, False, None
        if wait_for_ready and not self._client_ready_event.is_set():
            return True, False, self._append_pending_audio_event_locked(_FlushAudio())
        if self._audio_buffer:
            await self._queue.put(
                _AudioChunk(
                    audio=bytes(self._audio_buffer),
                    sample_rate=self._audio_buffer_sample_rate,
                    num_channels=self._audio_buffer_channels,
                )
            )
            self._audio_buffer.clear()
        if self._send_task is None:
            await self._drain_audio_queue()
            return False, False, None
        if (
            drop_if_not_ready
            and not wait_for_ready
            and not self._client_ready_event.is_set()
            and enqueue_flush
        ):
            await self._drain_audio_queue()
            return False, False, None
        if enqueue_flush:
            await self._queue.put(_FlushAudio())
        return False, enqueue_flush, None

    def _append_pending_audio_event_locked(
        self, event: _PendingAudioEvent
    ) -> ProtofaceException | None:
        event_bytes = len(event.audio) if isinstance(event, TTSAudioRawFrame) else 0
        if (
            len(self._pending_audio_events) >= _MAX_PENDING_AUDIO_EVENTS
            or self._pending_audio_bytes + event_bytes > _MAX_PENDING_AUDIO_BYTES
        ):
            return ProtofaceException(
                "Buffered Protoface TTS audio exceeded "
                f"{_MAX_PENDING_AUDIO_EVENTS} events or {_MAX_PENDING_AUDIO_BYTES} bytes."
            )
        self._pending_audio_events.append(event)
        self._pending_audio_bytes += event_bytes
        return None

    async def _handle_interruption(self) -> None:
        await self._interrupt_avatar_output(restart_send_task=True)

    async def _interrupt_avatar_output(self, *, restart_send_task: bool) -> None:
        async with self._audio_state_lock:
            self._audio_buffer.clear()
            self._pending_audio_events.clear()
            self._pending_audio_bytes = 0
            self._next_audio_send_at = 0.0
            self._should_measure_ttfb = False
            self._ttfb_metrics_active = False
            self._resampler = create_stream_resampler()
            await self._cancel_send_task()
            await self._drain_audio_queue()
        async with self._media_state_lock:
            self._media_generation += 1
            self._flushing_pending_media = False
            self._pending_media_frames.clear()
        with contextlib.suppress(Exception):
            await self._client.interrupt()
        self._client.clear_pending_media()
        if restart_send_task:
            await self._create_send_task()

    async def _process_pending_audio_events(self) -> None:
        async with self._audio_state_lock:
            while self._pending_audio_events:
                pending = self._pending_audio_events
                self._pending_audio_events = []
                self._pending_audio_bytes = 0
                for event in pending:
                    if self._fatal_error is not None:
                        await self._drain_audio_queue()
                        return
                    if isinstance(event, TTSAudioRawFrame):
                        await self._enqueue_audio_frame_locked(event, drop_if_not_ready=False)
                    else:
                        await self._flush_audio_locked(
                            wait_for_ready=False,
                            enqueue_flush=True,
                            drop_if_not_ready=False,
                        )
            self._client_ready_event.set()

    async def _create_send_task(self) -> None:
        if self._send_task is None or self._send_task.done():
            self._send_task = self.create_task(self._send_task_handler())

    async def _cancel_send_task(self) -> None:
        if self._send_task is not None:
            await self.cancel_task(self._send_task)
            self._send_task = None

    async def _drain_audio_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._queue.task_done()

    async def _send_task_handler(self) -> None:
        await self._client_ready_event.wait()
        if self._fatal_error is not None:
            await self._drain_audio_queue()
            return
        while True:
            item = await self._queue.get()
            try:
                if self._fatal_error is not None:
                    await self._drain_audio_queue()
                    return
                if isinstance(item, _AudioChunk):
                    await self._pace_audio_send(item)
                    await self._client.send_audio(
                        item.audio,
                        sample_rate=item.sample_rate,
                        num_channels=item.num_channels,
                    )
                    self._sent_audio_chunks += 1
                    if self._sent_audio_chunks == 1 or self._sent_audio_chunks % 100 == 0:
                        _debug_media(
                            "sent TTS audio chunks="
                            f"{self._sent_audio_chunks} bytes={len(item.audio)} "
                            f"sample_rate={item.sample_rate} channels={item.num_channels}"
                        )
                    if self._should_measure_ttfb:
                        await self.start_ttfb_metrics()
                        self._should_measure_ttfb = False
                        self._ttfb_metrics_active = True
                else:
                    await self._client.flush_audio()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._fail_fatal("Protoface avatar media send failed", exc)
                await self._drain_audio_queue()
                return
            finally:
                self._queue.task_done()

    async def _pace_audio_send(self, chunk: _AudioChunk) -> None:
        bytes_per_second = chunk.sample_rate * chunk.num_channels * _BYTES_PER_SAMPLE
        duration = len(chunk.audio) / bytes_per_second if bytes_per_second else 0.0
        send_ahead = max(0, cast(ProtofaceVideoSettings, self._settings).audio_send_ahead_ms) / 1000
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next_audio_send_at <= 0:
            self._next_audio_send_at = now

        lead = self._next_audio_send_at - now
        if lead > send_ahead:
            await asyncio.sleep(lead - send_ahead)
            now = loop.time()
        if self._next_audio_send_at < now:
            self._next_audio_send_at = now

        self._next_audio_send_at = max(now, self._next_audio_send_at) + duration

    async def _consume_media(self) -> None:
        try:
            async for frame in self._client.media_frames():
                buffered = False
                flush_generation: int | None = None
                overflow: ProtofaceException | None = None
                generation: int | None = None
                async with self._media_state_lock:
                    if self._fatal_error is not None:
                        self._client.clear_pending_media()
                        return
                    if not self._transport_ready or self._flushing_pending_media:
                        overflow = self._append_pending_media_frame_locked(frame)
                        buffered = True
                    elif self._pending_media_frames:
                        overflow = self._append_pending_media_frame_locked(frame)
                        if overflow is None:
                            self._flushing_pending_media = True
                            flush_generation = self._media_generation
                    else:
                        generation = self._media_generation
                if overflow is not None:
                    await self._fail_fatal("Protoface avatar media buffer overflow", overflow)
                    return
                if buffered:
                    continue
                if flush_generation is not None:
                    await self._flush_pending_media(flush_generation)
                    continue
                if generation is not None:
                    await self._push_media_frame(frame, generation=generation)
            if self._fatal_error is None:
                await self._fail_fatal(
                    "Protoface avatar media stream ended",
                    ProtofaceException("Protoface media stream ended unexpectedly."),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_fatal("Protoface avatar media stream failed", exc)

    def _append_pending_media_frame_locked(
        self, frame: _PendingMediaFrame
    ) -> ProtofaceException | None:
        if len(self._pending_media_frames) >= _MAX_PENDING_MEDIA_FRAMES:
            return ProtofaceException(
                f"Buffered Protoface avatar media exceeded {_MAX_PENDING_MEDIA_FRAMES} frames."
            )
        self._pending_media_frames.append(frame)
        return None

    async def _flush_pending_media(self, generation: int) -> None:
        try:
            while True:
                async with self._media_state_lock:
                    if self._fatal_error is not None or generation != self._media_generation:
                        return
                    if not self._pending_media_frames:
                        return
                    frame = self._pending_media_frames.popleft()
                try:
                    await self._push_media_frame(frame, generation=generation)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._fail_fatal("Protoface avatar media delivery failed", exc)
                    return
        finally:
            async with self._media_state_lock:
                if generation == self._media_generation:
                    self._flushing_pending_media = False

    async def _push_media_frame(
        self,
        frame: ProtofaceMediaFrame,
        *,
        generation: int | None = None,
    ) -> None:
        if self._should_drop_media(generation):
            return
        await self._stop_ttfb_metrics_if_active()
        if self._should_drop_media(generation):
            return
        if isinstance(frame, ProtofaceAudioFrame):
            await self._push_audio_frame(frame)
        else:
            await self._push_video_frame(frame)

    def _should_drop_media(self, generation: int | None) -> bool:
        return self._fatal_error is not None or (
            generation is not None and generation != self._media_generation
        )

    async def _stop_ttfb_metrics_if_active(self) -> None:
        if not self._ttfb_metrics_active:
            return
        self._ttfb_metrics_active = False
        await self.stop_ttfb_metrics()

    async def _push_audio_frame(self, frame: ProtofaceAudioFrame) -> None:
        self._pushed_audio_frames += 1
        if self._pushed_audio_frames == 1 or self._pushed_audio_frames % 100 == 0:
            _debug_media(
                "pushed Pipecat audio frames="
                f"{self._pushed_audio_frames} bytes={len(frame.audio)} "
                f"sample_rate={frame.sample_rate} channels={frame.num_channels}"
            )
        await self.push_frame(_to_pipecat_audio_frame(frame))

    async def _push_video_frame(self, frame: ProtofaceVideoFrame) -> None:
        self._pushed_video_frames += 1
        if self._pushed_video_frames == 1 or self._pushed_video_frames % 100 == 0:
            _debug_media(
                f"pushed Pipecat video frames={self._pushed_video_frames} size={frame.size}"
            )
        await self.push_frame(_to_pipecat_video_frame(frame))

    async def _fail_fatal(self, message: str, exc: Exception) -> None:
        self._fatal_error = exc
        async with self._audio_state_lock:
            self._audio_buffer.clear()
            self._pending_audio_events.clear()
            self._pending_audio_bytes = 0
            self._should_measure_ttfb = False
            self._ttfb_metrics_active = False
            self._resampler = create_stream_resampler()
            await self._drain_audio_queue()
        async with self._media_state_lock:
            self._media_generation += 1
            self._flushing_pending_media = False
            self._pending_media_frames.clear()
        self._client.clear_pending_media()
        with contextlib.suppress(Exception):
            await self._client.cancel()
        self._client_ready_event.set()
        if self._fatal_error_reported:
            return
        self._fatal_error_reported = True
        await self.push_error_frame(ErrorFrame(error=f"{message}: {exc}", fatal=True))


def _to_pipecat_audio_frame(frame: ProtofaceAudioFrame) -> SpeechOutputAudioRawFrame:
    out = SpeechOutputAudioRawFrame(
        audio=frame.audio,
        sample_rate=frame.sample_rate,
        num_channels=frame.num_channels,
    )
    if frame.transport_source is not None:
        out.transport_source = frame.transport_source
    return out


def _to_pipecat_video_frame(frame: ProtofaceVideoFrame) -> OutputImageRawFrame:
    out = OutputImageRawFrame(
        image=frame.image,
        size=frame.size,
        format=frame.format,
    )
    if frame.pts is not None:
        out.pts = frame.pts
    if frame.transport_source is not None:
        out.transport_source = frame.transport_source
    return out


__all__ = ["ProtofaceVideoService", "ProtofaceVideoSettings"]
