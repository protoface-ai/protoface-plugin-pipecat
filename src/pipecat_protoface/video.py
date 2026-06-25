"""Pipecat video service for Protoface avatars."""

from __future__ import annotations

import asyncio
import os
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
    ProtofaceMediaClient,
    ProtofaceRelayClient,
    ProtofaceVideoFrame,
)

_BYTES_PER_SAMPLE = 2
_DEFAULT_AUDIO_CHUNK_MS = 40


def _debug_media(message: str) -> None:
    if os.environ.get("PROTOFACE_DEBUG_MEDIA"):
        print(f"[pipecat-protoface] {message}", flush=True)


@dataclass
class ProtofaceVideoSettings(ServiceSettings):
    """Runtime settings for the Protoface Pipecat video service."""

    audio_chunk_ms: int = _DEFAULT_AUDIO_CHUNK_MS


@dataclass(slots=True)
class _AudioChunk:
    audio: bytes
    sample_rate: int
    num_channels: int


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
        self._queue: asyncio.Queue[_AudioChunk] = asyncio.Queue()
        self._connect_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._audio_task: asyncio.Task[None] | None = None
        self._video_task: asyncio.Task[None] | None = None
        self._transport_ready = False
        self._client_ready_event = asyncio.Event()
        self._should_measure_ttfb = False
        self._sent_audio_chunks = 0
        self._pushed_audio_frames = 0
        self._pushed_video_frames = 0

    def can_generate_metrics(self) -> bool:
        """Protoface can report TTFB through the Pipecat service hooks."""

        return True

    async def start(self, frame: StartFrame) -> None:
        """Start the hosted Protoface avatar session."""

        await super().start(frame)
        self._client_ready_event.clear()
        await self._create_send_task()
        self._connect_task = self.create_task(self._connect_client())

    async def stop(self, frame: EndFrame) -> None:
        """Stop the hosted Protoface avatar session."""

        await super().stop(frame)
        await self._cancel_connect_task()
        await self._flush_audio()
        await self._client.stop()
        await self._cancel_tasks()

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the hosted Protoface avatar session."""

        await super().cancel(frame)
        await self._cancel_connect_task()
        await self._client.cancel()
        await self._cancel_tasks()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process Pipecat frames through the Protoface avatar service."""

        await super().process_frame(frame, direction)
        if isinstance(frame, OutputTransportReadyFrame):
            self._transport_ready = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            self._should_measure_ttfb = True
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self.stop_ttfb_metrics()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._handle_audio_frame(frame)
        elif isinstance(frame, TTSStoppedFrame):
            await self._flush_audio()
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
            self._client_ready_event.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._client_ready_event.clear()
            await self.push_error_frame(
                ErrorFrame(error=f"Protoface avatar session failed to start: {exc}", fatal=True)
            )

    async def _create_consume_tasks(self) -> None:
        if self._audio_task is None:
            self._audio_task = self.create_task(self._consume_audio())
        if self._video_task is None:
            self._video_task = self.create_task(self._consume_video())

    async def _cancel_tasks(self) -> None:
        for attr in ("_send_task", "_audio_task", "_video_task"):
            task = getattr(self, attr)
            if task is not None:
                await self.cancel_task(task)
                setattr(self, attr, None)

    async def _cancel_connect_task(self) -> None:
        if self._connect_task is not None:
            await self.cancel_task(self._connect_task)
            self._connect_task = None

    async def _handle_audio_frame(self, frame: TTSAudioRawFrame) -> None:
        target_sample_rate = self._client.input_sample_rate or PROTOFACE_INPUT_SAMPLE_RATE
        if self._audio_buffer and (
            target_sample_rate != self._audio_buffer_sample_rate
            or frame.num_channels != self._audio_buffer_channels
        ):
            await self._flush_audio()
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

    async def _flush_audio(self) -> None:
        if self._audio_buffer:
            await self._queue.put(
                _AudioChunk(
                    audio=bytes(self._audio_buffer),
                    sample_rate=self._audio_buffer_sample_rate,
                    num_channels=self._audio_buffer_channels,
                )
            )
            self._audio_buffer.clear()
        await self._client.flush_audio()

    async def _handle_interruption(self) -> None:
        self._audio_buffer.clear()
        self._should_measure_ttfb = False
        await self._cancel_send_task()
        await self._drain_audio_queue()
        await self._client.interrupt()
        await self._create_send_task()

    async def _create_send_task(self) -> None:
        if self._send_task is None:
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
        while True:
            chunk = await self._queue.get()
            try:
                await self._client.send_audio(
                    chunk.audio,
                    sample_rate=chunk.sample_rate,
                    num_channels=chunk.num_channels,
                )
                self._sent_audio_chunks += 1
                if self._sent_audio_chunks == 1 or self._sent_audio_chunks % 100 == 0:
                    _debug_media(
                        "sent TTS audio chunks="
                        f"{self._sent_audio_chunks} bytes={len(chunk.audio)} "
                        f"sample_rate={chunk.sample_rate} channels={chunk.num_channels}"
                    )
                if self._should_measure_ttfb:
                    await self.start_ttfb_metrics()
                    self._should_measure_ttfb = False
            finally:
                self._queue.task_done()

    async def _consume_audio(self) -> None:
        async for frame in self._client.audio_frames():
            if self._transport_ready:
                self._pushed_audio_frames += 1
                if self._pushed_audio_frames == 1 or self._pushed_audio_frames % 100 == 0:
                    _debug_media(
                        "pushed Pipecat audio frames="
                        f"{self._pushed_audio_frames} bytes={len(frame.audio)} "
                        f"sample_rate={frame.sample_rate} channels={frame.num_channels}"
                    )
                await self.push_frame(_to_pipecat_audio_frame(frame))

    async def _consume_video(self) -> None:
        async for frame in self._client.video_frames():
            if self._transport_ready:
                self._pushed_video_frames += 1
                if self._pushed_video_frames == 1 or self._pushed_video_frames % 100 == 0:
                    _debug_media(
                        f"pushed Pipecat video frames={self._pushed_video_frames} size={frame.size}"
                    )
                await self.push_frame(_to_pipecat_video_frame(frame))


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
