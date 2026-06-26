"""Client interfaces for Protoface Pipecat media sessions."""

from __future__ import annotations

import asyncio
import contextlib
import json as jsonlib
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

import aiohttp
from PIL import Image
from pipecat_protoface._media import (
    MediaMessageType,
    decode_media_record,
    encode_media_record,
    media_audio_record,
)
from pipecat_protoface.version import __version__

DEFAULT_API_URL = "https://api.protoface.com"
PROTOFACE_INPUT_SAMPLE_RATE = 16_000
_USER_AGENT = f"pipecat-protoface/{__version__}"
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
_MAX_PENDING_KIND_FRAMES = 64


class ProtofaceException(Exception):
    """Raised for Protoface configuration or protocol errors."""


@dataclass(slots=True)
class ProtofaceAudioFrame:
    """PCM audio emitted by the Protoface avatar."""

    audio: bytes
    sample_rate: int
    num_channels: int = 1
    transport_source: str | None = None
    sequence_number: int | None = None


@dataclass(slots=True)
class ProtofaceVideoFrame:
    """Raw video frame emitted by the Protoface avatar."""

    image: bytes
    size: tuple[int, int]
    format: str = "RGB"
    pts: int | None = None
    transport_source: str | None = None
    sequence_number: int | None = None


ProtofaceMediaFrame = ProtofaceAudioFrame | ProtofaceVideoFrame


class ProtofaceMediaClient(Protocol):
    """Bidirectional media client used by ``ProtofaceVideoService``."""

    input_sample_rate: int

    async def start(
        self,
        *,
        avatar_id: str,
        max_duration_seconds: int | None = None,
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> str:
        """Start a hosted Protoface avatar session and return its id."""
        ...

    async def stop(self) -> None:
        """Gracefully end the hosted session."""
        ...

    async def cancel(self) -> None:
        """Immediately tear down the hosted session."""
        ...

    async def send_audio(self, audio: bytes, *, sample_rate: int, num_channels: int) -> None:
        """Send one chunk of TTS PCM to the avatar."""
        ...

    async def flush_audio(self) -> None:
        """Flush the current utterance."""
        ...

    async def interrupt(self) -> None:
        """Clear queued avatar speech after a user interruption."""
        ...

    def clear_pending_media(self) -> None:
        """Discard avatar media already queued locally."""
        ...

    def audio_frames(self) -> AsyncIterator[ProtofaceAudioFrame]:
        """Yield avatar speech audio frames."""
        ...

    def video_frames(self) -> AsyncIterator[ProtofaceVideoFrame]:
        """Yield avatar video frames."""
        ...

    def media_frames(self) -> AsyncIterator[ProtofaceMediaFrame]:
        """Yield avatar media frames in relay receive order."""
        ...


class ProtofaceRelayClient:
    """Client for Protoface Pipecat media sessions.

    ``send_audio()`` streams TTS PCM to Protoface. ``audio_frames()`` and
    ``video_frames()`` yield synchronized avatar media.
    """

    input_sample_rate = PROTOFACE_INPUT_SAMPLE_RATE

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Protoface API key is required.")
        self._api_key = api_key
        self._api_url = (api_url or os.environ.get("PROTOFACE_API_URL", DEFAULT_API_URL)).rstrip(
            "/"
        )
        self._session = session
        self._owns_session = session is None
        self._session_id: str | None = None
        self._media_ws: aiohttp.ClientWebSocketResponse | None = None
        self._media_task: asyncio.Task[None] | None = None
        self._media_error: Exception | None = None
        self._relay: dict[str, Any] | None = None
        self._audio_queue: asyncio.Queue[ProtofaceAudioFrame | None] = asyncio.Queue(
            maxsize=_MAX_PENDING_KIND_FRAMES
        )
        self._video_queue: asyncio.Queue[ProtofaceVideoFrame | None] = asyncio.Queue(
            maxsize=_MAX_PENDING_KIND_FRAMES
        )
        self._media_queue: asyncio.Queue[ProtofaceMediaFrame | None] = asyncio.Queue()
        self._media_sequence = 0

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def start(
        self,
        *,
        avatar_id: str,
        max_duration_seconds: int | None = None,
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> str:
        if (
            self._session_id is not None
            or self._media_ws is not None
            or self._media_task is not None
        ):
            await self.stop()
        self._media_error = None
        self._audio_queue = asyncio.Queue(maxsize=_MAX_PENDING_KIND_FRAMES)
        self._video_queue = asyncio.Queue(maxsize=_MAX_PENDING_KIND_FRAMES)
        self._media_queue = asyncio.Queue()
        self._media_sequence = 0
        payload: dict[str, Any] = {
            "avatar_id": avatar_id,
            "metadata": dict(metadata or {}),
            "relay_mode": "websocket",
        }
        if max_duration_seconds is not None:
            payload["max_duration_seconds"] = max_duration_seconds

        response = await self._json("POST", "/v1/pipecat/sessions", json=payload)
        session = response.get("session")
        relay = response.get("relay")
        if not isinstance(session, dict) or not isinstance(relay, dict):
            raise ProtofaceException("Protoface API returned an invalid Pipecat session.")

        session_id = session.get("id")
        if not isinstance(session_id, str):
            raise ProtofaceException("Protoface API response missing session id.")
        self._session_id = session_id
        self._relay = relay
        self.input_sample_rate = int(relay.get("audio_sample_rate") or PROTOFACE_INPUT_SAMPLE_RATE)

        try:
            relay_type = str(relay.get("type") or "websocket")
            if relay_type == "websocket":
                await self._connect_media_websocket(relay)
                return session_id
            raise ProtofaceException(f"Unsupported Protoface Pipecat relay type: {relay_type!r}")
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self.stop()
            raise
        except Exception:
            with contextlib.suppress(Exception):
                await self.stop()
            raise

    async def stop(self) -> None:
        await self.flush_audio()
        if self._session_id is not None:
            with contextlib.suppress(Exception):
                await self._json("POST", f"/v1/sessions/{self._session_id}/end")
        await self._close_media_websocket()
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None
        self._session_id = None

    async def cancel(self) -> None:
        await self.stop()

    async def send_audio(self, audio: bytes, *, sample_rate: int, num_channels: int) -> None:
        if not audio:
            return
        media_ws = self._media_ws
        if media_ws is not None:
            await media_ws.send_bytes(
                media_audio_record(audio, sample_rate=sample_rate, num_channels=num_channels)
            )
            return
        raise ProtofaceException("Protoface media socket is not connected.")

    async def flush_audio(self) -> None:
        media_ws = self._media_ws
        if media_ws is not None:
            with contextlib.suppress(Exception):
                await media_ws.send_bytes(encode_media_record(MediaMessageType.FLUSH))

    async def interrupt(self) -> None:
        media_ws = self._media_ws
        if media_ws is not None:
            with contextlib.suppress(Exception):
                await media_ws.send_bytes(encode_media_record(MediaMessageType.INTERRUPT))
            return

    def clear_pending_media(self) -> None:
        self._drain_queue(self._audio_queue)
        self._drain_queue(self._video_queue)
        self._drain_queue(self._media_queue)

    async def _audio_frame_iterator(self) -> AsyncIterator[ProtofaceAudioFrame]:
        while True:
            frame = await self._audio_queue.get()
            if frame is None:
                if self._media_error is not None:
                    raise self._media_error
                return
            yield frame

    async def _video_frame_iterator(self) -> AsyncIterator[ProtofaceVideoFrame]:
        while True:
            frame = await self._video_queue.get()
            if frame is None:
                if self._media_error is not None:
                    raise self._media_error
                return
            yield frame

    async def _media_frame_iterator(self) -> AsyncIterator[ProtofaceMediaFrame]:
        while True:
            frame = await self._media_queue.get()
            if frame is None:
                if self._media_error is not None:
                    raise self._media_error
                return
            yield frame

    def audio_frames(self) -> AsyncIterator[ProtofaceAudioFrame]:
        return self._audio_frame_iterator()

    def video_frames(self) -> AsyncIterator[ProtofaceVideoFrame]:
        return self._video_frame_iterator()

    def media_frames(self) -> AsyncIterator[ProtofaceMediaFrame]:
        return self._media_frame_iterator()

    async def _connect_media_websocket(self, relay: Mapping[str, Any]) -> None:
        media_url = str(relay["media_url"])
        media_token = str(relay["media_token"])
        protocol = str(relay.get("protocol") or "protoface.pipecat.media.v1")
        self._media_ws = await self._ensure_session().ws_connect(
            media_url,
            headers={
                "Authorization": f"Bearer {media_token}",
                "User-Agent": _USER_AGENT,
            },
            protocols=(protocol, media_token),
        )
        self._media_task = asyncio.create_task(self._consume_media_websocket())

    async def _consume_media_websocket(self) -> None:
        ws = self._media_ws
        if ws is None:
            return
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_media_record(bytes(msg.data))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    ws_error = ws.exception()
                    if ws_error is not None:
                        raise ProtofaceException(f"Protoface media socket error: {ws_error}")
                    raise ProtofaceException("Protoface media socket error.")
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._media_error = exc
        finally:
            self._put_bounded(self._audio_queue, None)
            self._put_bounded(self._video_queue, None)
            await self._media_queue.put(None)

    async def _handle_media_record(self, data: bytes) -> None:
        record = decode_media_record(data)
        if record.msg_type is MediaMessageType.AUDIO:
            audio_frame = ProtofaceAudioFrame(
                audio=record.blob,
                sample_rate=int(record.header.get("sample_rate") or self.input_sample_rate),
                num_channels=int(record.header.get("num_channels") or 1),
                transport_source="protoface-direct",
                sequence_number=self._next_media_sequence(),
            )
            self._put_bounded(self._audio_queue, audio_frame)
            await self._media_queue.put(audio_frame)
            return
        if record.msg_type is MediaMessageType.VIDEO:
            video_frame = _decode_video_record(
                record.header,
                record.blob,
                sequence_number=self._next_media_sequence(),
            )
            self._put_bounded(self._video_queue, video_frame)
            await self._media_queue.put(video_frame)
            return
        if record.msg_type is MediaMessageType.ERROR:
            message = record.header.get("message") or "Protoface media error"
            raise ProtofaceException(str(message))

    def _next_media_sequence(self) -> int:
        sequence_number = self._media_sequence
        self._media_sequence += 1
        return sequence_number

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

    @staticmethod
    def _put_bounded(queue: asyncio.Queue[Any], item: Any) -> None:
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(item)

    async def _close_media_websocket(self) -> None:
        task = self._media_task
        self._media_task = None
        ws = self._media_ws
        self._media_ws = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_DEFAULT_REQUEST_TIMEOUT_SECONDS)
            )
        return self._session

    async def _json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        }
        async with self._ensure_session().request(
            method,
            f"{self._api_url}{path}",
            json=json,
            headers=headers,
        ) as response:
            payload = await _read_payload(response)
            if response.ok and isinstance(payload, dict):
                return payload
            if response.ok:
                raise ProtofaceException("Protoface API returned a non-object JSON response.")
            raise ProtofaceException(f"Protoface API returned {response.status}: {payload!r}")


async def _read_payload(response: aiohttp.ClientResponse) -> object:
    text = await response.text()
    if not text:
        return {}
    try:
        return jsonlib.loads(text)
    except ValueError:
        return {"raw": text}


def _decode_video_record(
    header: Mapping[str, Any],
    blob: bytes,
    *,
    sequence_number: int | None = None,
) -> ProtofaceVideoFrame:
    encoding = str(header.get("encoding") or "rgb24")
    pts_raw = header.get("timestamp_ms")
    pts = int(pts_raw) if pts_raw is not None else None
    if encoding == "jpeg":
        image = Image.open(BytesIO(blob)).convert("RGB")
        return ProtofaceVideoFrame(
            image=image.tobytes(),
            size=image.size,
            format="RGB",
            pts=pts,
            transport_source="protoface-direct",
            sequence_number=sequence_number,
        )
    if encoding == "rgb24":
        width = int(header["width"])
        height = int(header["height"])
        return ProtofaceVideoFrame(
            image=blob,
            size=(width, height),
            format="RGB",
            pts=pts,
            transport_source="protoface-direct",
            sequence_number=sequence_number,
        )
    raise ProtofaceException(f"Unsupported Protoface video encoding: {encoding!r}")


__all__ = [
    "DEFAULT_API_URL",
    "PROTOFACE_INPUT_SAMPLE_RATE",
    "ProtofaceAudioFrame",
    "ProtofaceException",
    "ProtofaceMediaClient",
    "ProtofaceMediaFrame",
    "ProtofaceRelayClient",
    "ProtofaceVideoFrame",
]
