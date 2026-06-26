"""Protoface video avatar service for Pipecat."""

from .video import ProtofaceVideoService, ProtofaceVideoSettings
from .version import __version__

__all__ = [
    "ProtofaceVideoService",
    "ProtofaceVideoSettings",
    "__version__",
]
