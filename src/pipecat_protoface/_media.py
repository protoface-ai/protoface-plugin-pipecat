"""Binary media framing for Protoface Pipecat relay sessions."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

_U32 = struct.Struct(">I")
_MAX_RECORD_BYTES = 64 * 1024 * 1024


class MediaMessageType(IntEnum):
    """Media message kinds."""

    AUDIO = 0x01
    FLUSH = 0x02
    INTERRUPT = 0x03
    LISTEN_AUDIO = 0x04
    CLOSE = 0x05

    READY = 0x10
    VIDEO = 0x11
    METRICS = 0x12
    ERROR = 0x13


@dataclass(frozen=True, slots=True)
class MediaRecord:
    """One media record."""

    msg_type: MediaMessageType
    header: dict[str, Any]
    blob: bytes = b""


def encode_media_record(
    msg_type: MediaMessageType,
    header: dict[str, Any] | None = None,
    blob: bytes = b"",
) -> bytes:
    """Serialize one media record."""

    hdr_bytes = json.dumps(header or {}, separators=(",", ":")).encode("utf-8")
    body = struct.pack(">BI", int(msg_type), len(hdr_bytes)) + hdr_bytes + blob
    if len(body) > _MAX_RECORD_BYTES:
        raise ValueError(
            f"media record too large: {len(body)} bytes exceeds max {_MAX_RECORD_BYTES}"
        )
    return _U32.pack(len(body)) + body


def decode_media_record(data: bytes) -> MediaRecord:
    """Parse one complete media record."""

    if len(data) < 4:
        raise ValueError(f"media record too short: {len(data)} bytes")
    (total_len,) = _U32.unpack(data[:4])
    if total_len > _MAX_RECORD_BYTES:
        raise ValueError(
            f"media record too large: length prefix {total_len} exceeds max {_MAX_RECORD_BYTES}"
        )
    if len(data) != 4 + total_len:
        raise ValueError(f"media record length mismatch: expected {4 + total_len}, got {len(data)}")
    body = data[4:]
    if len(body) < 5:
        raise ValueError(f"media record body too short: {len(body)} bytes")
    msg_type = MediaMessageType(body[0])
    (hdr_len,) = _U32.unpack(body[1:5])
    hdr_end = 5 + hdr_len
    if len(body) < hdr_end:
        raise ValueError(f"media record header truncated: need {hdr_end}, have {len(body)}")
    header = json.loads(body[5:hdr_end].decode("utf-8")) if hdr_len else {}
    if not isinstance(header, dict):
        raise ValueError("media record header must decode to an object")
    return MediaRecord(msg_type=msg_type, header=header, blob=body[hdr_end:])


def media_audio_record(
    audio: bytes,
    *,
    sample_rate: int,
    num_channels: int,
    source: str | None = None,
) -> bytes:
    """Build a PCM16 audio record."""

    header: dict[str, Any] = {
        "sample_rate": sample_rate,
        "num_channels": num_channels,
    }
    if source is not None:
        header["source"] = source
    return encode_media_record(MediaMessageType.AUDIO, header, audio)


def media_video_record(
    image: bytes,
    *,
    width: int,
    height: int,
    encoding: str,
    frame_index: int | None = None,
    timestamp_ms: int | None = None,
) -> bytes:
    """Build an encoded video frame record."""

    header: dict[str, Any] = {
        "width": width,
        "height": height,
        "encoding": encoding,
    }
    if frame_index is not None:
        header["frame_index"] = frame_index
    if timestamp_ms is not None:
        header["timestamp_ms"] = timestamp_ms
    return encode_media_record(MediaMessageType.VIDEO, header, image)
