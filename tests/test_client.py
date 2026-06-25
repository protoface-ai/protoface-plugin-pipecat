from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pytest
from pipecat_protoface._client import ProtofaceRelayClient
from pipecat_protoface._media import (
    MediaMessageType,
    decode_media_record,
    media_audio_record,
    media_video_record,
)
from pipecat_protoface.version import __version__


class _FakeWSMessage:
    def __init__(self, data: bytes) -> None:
        self.type = aiohttp.WSMsgType.BINARY
        self.data = data


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self.messages: asyncio.Queue[_FakeWSMessage | None] = asyncio.Queue()

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True
        await self.messages.put(None)

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> _FakeWSMessage:
        item = await self.messages.get()
        if item is None:
            raise StopAsyncIteration
        return item


class _FakeWebSocketSession:
    def __init__(self) -> None:
        self.ws = _FakeWebSocket()
        self.ws_connects: list[dict[str, Any]] = []
        self.closed = False

    async def ws_connect(
        self,
        url: str,
        *,
        headers: dict[str, str],
        protocols: tuple[str, ...],
    ) -> _FakeWebSocket:
        self.ws_connects.append({"url": url, "headers": headers, "protocols": protocols})
        return self.ws

    async def close(self) -> None:
        self.closed = True


class _TestDirectRelayClient(ProtofaceRelayClient):
    def __init__(self, session: _FakeWebSocketSession) -> None:
        super().__init__(
            api_key="sk_test",
            session=session,  # type: ignore[arg-type]
        )
        self.requests: list[dict[str, Any]] = []

    async def _json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.requests.append({"method": method, "path": path, "json": json})
        if path == "/v1/pipecat/sessions":
            return {
                "session": {"id": "sess_test"},
                "relay": {
                    "type": "websocket",
                    "protocol": "protoface.pipecat.media.v1",
                    "media_url": "ws://api.test/v1/pipecat/sessions/sess_test/media",
                    "media_token": "pfm_client",
                    "audio_sample_rate": 16000,
                    "video_encoding": "rgb24",
                    "decoded_video_format": "RGB",
                    "supports_audio_output": True,
                },
            }
        if path == "/v1/sessions/sess_test/end":
            return {}
        raise AssertionError(f"unexpected request: {method} {path}")


def test_relay_client_uses_protoface_api_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROTOFACE_API_URL", "http://localhost:8000/")

    client = ProtofaceRelayClient(api_key="sk_test")

    assert client._api_url == "http://localhost:8000"


@pytest.mark.asyncio
async def test_direct_relay_client_sends_audio_control_records() -> None:
    session = _FakeWebSocketSession()
    client = _TestDirectRelayClient(session)

    await client.start(avatar_id="av_demo")

    assert client.requests[0]["json"]["relay_mode"] == "websocket"
    assert session.ws_connects == [
        {
            "url": "ws://api.test/v1/pipecat/sessions/sess_test/media",
            "headers": {
                "Authorization": "Bearer pfm_client",
                "User-Agent": f"pipecat-protoface/{__version__}",
            },
            "protocols": ("protoface.pipecat.media.v1", "pfm_client"),
        }
    ]

    await client.send_audio(b"pcm", sample_rate=16000, num_channels=1)
    await client.flush_audio()
    await client.interrupt()

    audio = decode_media_record(session.ws.sent[0])
    assert audio.msg_type is MediaMessageType.AUDIO
    assert audio.header == {"sample_rate": 16000, "num_channels": 1}
    assert audio.blob == b"pcm"
    assert decode_media_record(session.ws.sent[1]).msg_type is MediaMessageType.FLUSH
    assert decode_media_record(session.ws.sent[2]).msg_type is MediaMessageType.FLUSH
    assert decode_media_record(session.ws.sent[3]).msg_type is MediaMessageType.INTERRUPT

    await client.stop()
    assert session.ws.closed


@pytest.mark.asyncio
async def test_direct_relay_client_consumes_audio_and_video_records() -> None:
    session = _FakeWebSocketSession()
    client = _TestDirectRelayClient(session)

    await client.start(avatar_id="av_demo")
    await session.ws.messages.put(
        _FakeWSMessage(media_audio_record(b"out", sample_rate=16000, num_channels=1))
    )
    await session.ws.messages.put(
        _FakeWSMessage(
            media_video_record(
                b"\x00\x01\x02\x03\x04\x05",
                width=1,
                height=2,
                encoding="rgb24",
                frame_index=1,
                timestamp_ms=123,
            )
        )
    )

    audio_iter = client.audio_frames()
    video_iter = client.video_frames()
    audio = await audio_iter.__anext__()
    video = await video_iter.__anext__()

    assert audio.audio == b"out"
    assert audio.sample_rate == 16000
    assert audio.transport_source == "protoface-direct"
    assert video.image == b"\x00\x01\x02\x03\x04\x05"
    assert video.size == (1, 2)
    assert video.pts == 123
    assert video.transport_source == "protoface-direct"

    await client.stop()
