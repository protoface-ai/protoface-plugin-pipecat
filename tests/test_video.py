from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Coroutine, Mapping
from typing import Any

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
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
from pipecat_protoface import ProtofaceVideoService, ProtofaceVideoSettings
from pipecat_protoface._client import (
    ProtofaceAudioFrame,
    ProtofaceMediaClient,
    ProtofaceMediaFrame,
    ProtofaceVideoFrame,
)
from pipecat_protoface.video import (
    _MAX_PENDING_MEDIA_FRAMES,
    _to_pipecat_audio_frame,
    _to_pipecat_video_frame,
)


class FakeMediaClient:
    input_sample_rate = 16_000

    def __init__(
        self,
        *,
        start_delay: float = 0.0,
        start_error: Exception | None = None,
        send_error: Exception | None = None,
        audio_error: Exception | None = None,
        session_input_sample_rate: int | None = None,
    ) -> None:
        self.input_sample_rate = 16_000
        self.start_delay = start_delay
        self.start_error = start_error
        self.send_error = send_error
        self.audio_error = audio_error
        self.session_input_sample_rate = session_input_sample_rate
        self.started: dict[str, object] | None = None
        self.sent_audio: list[tuple[bytes, int, int]] = []
        self.sent_at: list[float] = []
        self.events: list[str] = []
        self.starts = 0
        self.flushed = 0
        self.interrupted = 0
        self.stopped = 0
        self.canceled = 0
        self.media_reads_enabled = asyncio.Event()
        self.media_reads_enabled.set()
        self._audio: asyncio.Queue[ProtofaceAudioFrame | None] = asyncio.Queue()
        self._video: asyncio.Queue[ProtofaceVideoFrame | None] = asyncio.Queue()
        self._media: asyncio.Queue[ProtofaceMediaFrame | None] = asyncio.Queue()

    async def start(
        self,
        *,
        avatar_id: str,
        max_duration_seconds: int | None = None,
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> str:
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        if self.start_error is not None:
            raise self.start_error
        self.starts += 1
        if self.session_input_sample_rate is not None:
            self.input_sample_rate = self.session_input_sample_rate
        self._audio = asyncio.Queue()
        self._video = asyncio.Queue()
        self._media = asyncio.Queue()
        self.started = {
            "avatar_id": avatar_id,
            "max_duration_seconds": max_duration_seconds,
            "metadata": dict(metadata or {}),
        }
        return "sess_test"

    async def stop(self) -> None:
        self.stopped += 1

    async def cancel(self) -> None:
        self.canceled += 1

    async def send_audio(self, audio: bytes, *, sample_rate: int, num_channels: int) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent_at.append(asyncio.get_running_loop().time())
        self.events.append("send")
        self.sent_audio.append((audio, sample_rate, num_channels))

    async def flush_audio(self) -> None:
        self.events.append("flush")
        self.flushed += 1

    async def interrupt(self) -> None:
        self.interrupted += 1

    def clear_pending_media(self) -> None:
        self._drain_queue(self._audio)
        self._drain_queue(self._video)
        self._drain_queue(self._media)

    async def push_audio(self, frame: ProtofaceAudioFrame) -> None:
        await self._audio.put(frame)
        await self._media.put(frame)

    async def push_video(self, frame: ProtofaceVideoFrame) -> None:
        await self._video.put(frame)
        await self._media.put(frame)

    async def close_streams(self) -> None:
        await self._audio.put(None)
        await self._video.put(None)
        await self._media.put(None)

    async def audio_frames(self) -> AsyncIterator[ProtofaceAudioFrame]:
        if self.audio_error is not None:
            raise self.audio_error
        while True:
            frame = await self._audio.get()
            if frame is None:
                return
            yield frame

    async def video_frames(self) -> AsyncIterator[ProtofaceVideoFrame]:
        while True:
            frame = await self._video.get()
            if frame is None:
                return
            yield frame

    async def media_frames(self) -> AsyncIterator[ProtofaceMediaFrame]:
        if self.audio_error is not None:
            raise self.audio_error
        while True:
            await self.media_reads_enabled.wait()
            frame = await self._media.get()
            if frame is None:
                return
            yield frame

    @staticmethod
    def _drain_queue(queue: asyncio.Queue[Any]) -> None:
        saw_terminal = False
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                saw_terminal = True
        if saw_terminal:
            queue.put_nowait(None)


class TestableProtofaceVideoService(ProtofaceVideoService):
    __test__ = False

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.pushed: list[object] = []
        self.started_ttfb = 0
        self.stopped_ttfb = 0

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *args: object,
        **kwargs: object,
    ) -> asyncio.Task[Any]:
        del args, kwargs
        return asyncio.create_task(coroutine)

    async def cancel_task(self, task: asyncio.Task[Any], timeout: float | None = 1.0) -> None:
        del timeout
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def push_frame(
        self,
        frame: object,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        del direction
        self.pushed.append(frame)

    async def push_error_frame(self, frame: ErrorFrame) -> None:
        self.pushed.append(frame)

    async def start_ttfb_metrics(self, *, start_time: float | None = None) -> None:
        del start_time
        self.started_ttfb += 1

    async def stop_ttfb_metrics(self, *, end_time: float | None = None) -> None:
        del end_time
        self.stopped_ttfb += 1


async def _wait_until(predicate: object, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():  # type: ignore[operator]
            return
        await asyncio.sleep(0.01)
    assert predicate()  # type: ignore[operator]


def test_fake_client_satisfies_protocol() -> None:
    client: ProtofaceMediaClient = FakeMediaClient()
    assert client.input_sample_rate == 16_000


def test_audio_frame_conversion() -> None:
    frame = _to_pipecat_audio_frame(
        ProtofaceAudioFrame(
            audio=b"\x00\x01",
            sample_rate=16_000,
            num_channels=1,
            transport_source="protoface",
        )
    )

    assert isinstance(frame, SpeechOutputAudioRawFrame)
    assert frame.audio == b"\x00\x01"
    assert frame.sample_rate == 16_000
    assert frame.num_channels == 1
    assert frame.transport_source == "protoface"


def test_video_frame_conversion() -> None:
    frame = _to_pipecat_video_frame(
        ProtofaceVideoFrame(
            image=b"rgb",
            size=(1, 1),
            format="RGB",
            pts=123,
            transport_source="protoface",
        )
    )

    assert isinstance(frame, OutputImageRawFrame)
    assert frame.image == b"rgb"
    assert frame.size == (1, 1)
    assert frame.format == "RGB"
    assert frame.pts == 123
    assert frame.transport_source == "protoface"


@pytest.mark.asyncio
async def test_tts_audio_frame_shape_matches_pipecat() -> None:
    frame = TTSAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16_000, num_channels=1)
    assert frame.num_frames == 160


@pytest.mark.asyncio
async def test_service_starts_and_stops_media_client() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        max_duration_seconds=120,
        metadata={"customer_session_id": "abc"},
        media_client=client,
    )

    await service.start(StartFrame())
    await _wait_until(lambda: client.started is not None)
    assert client.started == {
        "avatar_id": "av_demo",
        "max_duration_seconds": 120,
        "metadata": {"customer_session_id": "abc"},
    }

    await client.close_streams()
    await service.stop(EndFrame())
    assert client.flushed == 1
    assert client.stopped == 1


@pytest.mark.asyncio
async def test_service_emits_avatar_media_after_transport_ready() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(OutputTransportReadyFrame(), FrameDirection.DOWNSTREAM)
    await _wait_until(lambda: client.started is not None)
    await client.push_audio(ProtofaceAudioFrame(audio=b"\x00\x01", sample_rate=16_000))
    await client.push_video(ProtofaceVideoFrame(image=b"rgb", size=(1, 1), pts=12))

    await _wait_until(
        lambda: (
            any(isinstance(frame, SpeechOutputAudioRawFrame) for frame in service.pushed)
            and any(isinstance(frame, OutputImageRawFrame) for frame in service.pushed)
        )
    )
    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_buffers_avatar_media_until_transport_ready() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await _wait_until(lambda: client.started is not None)
    await client.push_audio(ProtofaceAudioFrame(audio=b"\x00\x01", sample_rate=16_000))
    await client.push_video(ProtofaceVideoFrame(image=b"rgb", size=(1, 1), pts=12))
    await asyncio.sleep(0.05)

    assert not any(isinstance(frame, SpeechOutputAudioRawFrame) for frame in service.pushed)
    assert not any(isinstance(frame, OutputImageRawFrame) for frame in service.pushed)

    await service.process_frame(OutputTransportReadyFrame(), FrameDirection.DOWNSTREAM)
    await _wait_until(
        lambda: (
            any(isinstance(frame, SpeechOutputAudioRawFrame) for frame in service.pushed)
            and any(isinstance(frame, OutputImageRawFrame) for frame in service.pushed)
        )
    )
    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_stops_ttfb_on_first_avatar_media() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(OutputTransportReadyFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 640, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await _wait_until(lambda: service.started_ttfb == 1)

    assert service.stopped_ttfb == 0
    assert any(isinstance(frame, TTSStartedFrame) for frame in service.pushed)
    assert any(isinstance(frame, BotStartedSpeakingFrame) for frame in service.pushed)

    await client.push_audio(ProtofaceAudioFrame(audio=b"\x00\x01", sample_rate=16_000))
    await _wait_until(lambda: service.stopped_ttfb == 1)

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_replays_buffered_avatar_media_in_received_order() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await _wait_until(lambda: client.started is not None)
    await client.push_video(ProtofaceVideoFrame(image=b"first", size=(1, 1), pts=1))
    await _wait_until(lambda: len(service._pending_media_frames) == 1)
    await client.push_audio(ProtofaceAudioFrame(audio=b"second", sample_rate=16_000))
    await _wait_until(lambda: len(service._pending_media_frames) == 2)

    await service.process_frame(OutputTransportReadyFrame(), FrameDirection.DOWNSTREAM)

    media_frames = [
        frame
        for frame in service.pushed
        if isinstance(frame, OutputImageRawFrame | SpeechOutputAudioRawFrame)
    ]
    assert [type(frame) for frame in media_frames] == [
        OutputImageRawFrame,
        SpeechOutputAudioRawFrame,
    ]

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_keeps_buffered_media_before_new_ready_media() -> None:
    client = FakeMediaClient()

    class OrderingProbeService(TestableProtofaceVideoService):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.injected_new_media = False

        async def push_frame(
            self,
            frame: object,
            direction: FrameDirection = FrameDirection.DOWNSTREAM,
        ) -> None:
            if (
                isinstance(frame, OutputImageRawFrame)
                and frame.image == b"old"
                and not self.injected_new_media
            ):
                self.injected_new_media = True
                await client.push_video(ProtofaceVideoFrame(image=b"new", size=(1, 1), pts=2))
                await asyncio.sleep(0.05)
            await super().push_frame(frame, direction)

    service = OrderingProbeService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await _wait_until(lambda: client.started is not None)
    await client.push_video(ProtofaceVideoFrame(image=b"old", size=(1, 1), pts=1))
    await _wait_until(lambda: len(service._pending_media_frames) == 1)

    await service.process_frame(OutputTransportReadyFrame(), FrameDirection.DOWNSTREAM)
    await _wait_until(
        lambda: (
            len([frame for frame in service.pushed if isinstance(frame, OutputImageRawFrame)]) == 2
        )
    )

    video_frames = [frame for frame in service.pushed if isinstance(frame, OutputImageRawFrame)]
    assert [frame.image for frame in video_frames] == [b"old", b"new"]

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_caps_buffered_avatar_media_until_transport_ready() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await _wait_until(lambda: client.started is not None)
    for index in range(_MAX_PENDING_MEDIA_FRAMES + 10):
        await client.push_video(ProtofaceVideoFrame(image=b"rgb", size=(1, 1), pts=index))

    def pending_video_pts() -> list[int | None]:
        return [
            frame.pts
            for frame in service._pending_media_frames
            if isinstance(frame, ProtofaceVideoFrame)
        ]

    await _wait_until(
        lambda: (
            len(pending_video_pts()) == _MAX_PENDING_MEDIA_FRAMES
            and pending_video_pts()[-1] == _MAX_PENDING_MEDIA_FRAMES + 9
        )
    )

    await service.process_frame(OutputTransportReadyFrame(), FrameDirection.DOWNSTREAM)
    video_frames = [frame for frame in service.pushed if isinstance(frame, OutputImageRawFrame)]
    assert len(video_frames) == _MAX_PENDING_MEDIA_FRAMES
    assert video_frames[0].pts == 10
    assert video_frames[-1].pts == _MAX_PENDING_MEDIA_FRAMES + 9

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_chunks_audio_and_interrupts() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 640, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await _wait_until(lambda: bool(client.sent_audio))

    assert client.sent_audio == [(b"\x00\x00" * 640, 16_000, 1)]

    await service.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert client.interrupted == 1

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_drops_queued_avatar_media_on_interruption() -> None:
    client = FakeMediaClient()
    client.media_reads_enabled.clear()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(OutputTransportReadyFrame(), FrameDirection.DOWNSTREAM)
    await _wait_until(lambda: client.started is not None)
    await client.push_audio(ProtofaceAudioFrame(audio=b"stale", sample_rate=16_000))
    await client.push_video(ProtofaceVideoFrame(image=b"stale", size=(1, 1), pts=1))

    await service.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    client.media_reads_enabled.set()
    await asyncio.sleep(0.05)

    assert client.interrupted == 1
    assert not any(isinstance(frame, SpeechOutputAudioRawFrame) for frame in service.pushed)
    assert not any(isinstance(frame, OutputImageRawFrame) for frame in service.pushed)

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_paces_queued_audio_sends() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
        settings=ProtofaceVideoSettings(model=None, audio_send_ahead_ms=0),
    )

    await service.start(StartFrame())
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 1280, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )

    await _wait_until(lambda: len(client.sent_audio) == 2)

    assert client.sent_at[1] - client.sent_at[0] >= 0.025

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_buffers_tts_until_media_client_is_ready() -> None:
    client = FakeMediaClient(start_delay=0.05)
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 640, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )

    assert client.sent_audio == []
    await _wait_until(lambda: bool(client.sent_audio))
    assert client.sent_audio == [(b"\x00\x00" * 640, 16_000, 1)]

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_resamples_queued_tts_to_negotiated_sample_rate() -> None:
    client = FakeMediaClient(start_delay=0.05, session_input_sample_rate=24_000)
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 1280, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await service.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

    assert client.sent_audio == []
    await _wait_until(lambda: bool(client.sent_audio))
    assert client.sent_audio[0][1:] == (24_000, 1)

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_second_start_replaces_existing_session() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await _wait_until(lambda: client.starts == 1)

    await service.start(StartFrame())
    await _wait_until(lambda: client.starts == 2)

    assert client.canceled == 1

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_start_failure_unblocks_send_task() -> None:
    client = FakeMediaClient(start_error=RuntimeError("start failed"))
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 640, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await _wait_until(lambda: any(isinstance(frame, ErrorFrame) for frame in service.pushed))
    await asyncio.wait_for(
        service.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM),
        timeout=1.0,
    )

    assert client.sent_audio == []
    assert client.canceled == 1

    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_restarts_after_fatal_send_error() -> None:
    client = FakeMediaClient(send_error=RuntimeError("send failed"))
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 640, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await _wait_until(lambda: any(isinstance(frame, ErrorFrame) for frame in service.pushed))

    client.send_error = None
    await service.start(StartFrame())
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x01\x01" * 640, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )

    await _wait_until(lambda: client.sent_audio == [(b"\x01\x01" * 640, 16_000, 1)])

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_send_errors_push_fatal_error() -> None:
    client = FakeMediaClient(send_error=RuntimeError("send failed"))
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(OutputTransportReadyFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 640, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )

    await _wait_until(lambda: any(isinstance(frame, ErrorFrame) for frame in service.pushed))
    error = next(frame for frame in service.pushed if isinstance(frame, ErrorFrame))
    assert error.fatal is True
    assert "send failed" in error.error

    await client.push_audio(ProtofaceAudioFrame(audio=b"late", sample_rate=16_000))
    await asyncio.sleep(0.05)
    assert not any(isinstance(frame, SpeechOutputAudioRawFrame) for frame in service.pushed)

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_media_errors_push_fatal_error() -> None:
    client = FakeMediaClient(audio_error=RuntimeError("media failed"))
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())

    await _wait_until(lambda: any(isinstance(frame, ErrorFrame) for frame in service.pushed))
    error = next(frame for frame in service.pushed if isinstance(frame, ErrorFrame))
    assert error.fatal is True
    assert "media failed" in error.error

    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 640, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.05)
    assert client.sent_audio == []

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_cancel_clears_queued_tts_audio() -> None:
    client = FakeMediaClient(start_delay=0.05)
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 640, sample_rate=16_000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await service.cancel(CancelFrame())

    client.start_delay = 0
    await service.start(StartFrame())
    await _wait_until(lambda: client.started is not None)
    await asyncio.sleep(0.05)

    assert client.sent_audio == []

    await client.close_streams()
    await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_service_flushes_leftover_audio_with_original_channels() -> None:
    client = FakeMediaClient()
    service = TestableProtofaceVideoService(
        api_key="sk_test",
        avatar_id="av_demo",
        media_client=client,
    )

    await service.start(StartFrame())
    await service.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 160 * 2, sample_rate=16_000, num_channels=2),
        FrameDirection.DOWNSTREAM,
    )
    assert client.sent_audio == []

    await service.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    await _wait_until(lambda: bool(client.sent_audio))

    assert client.sent_audio == [(b"\x00\x00" * 160 * 2, 16_000, 2)]
    assert client.events == ["send", "flush"]
    assert any(isinstance(frame, TTSStoppedFrame) for frame in service.pushed)

    await client.close_streams()
    await service.cancel(CancelFrame())
