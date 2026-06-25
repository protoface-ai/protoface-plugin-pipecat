# Protoface Pipecat Plugin

[![PyPI - Version](https://img.shields.io/pypi/v/pipecat-protoface)](https://pypi.python.org/pypi/pipecat-protoface)

Add Protoface video avatars to Pipecat pipelines.

Maintained by [Protoface](https://protoface.com).

## Installation

```bash
pip install pipecat-protoface
```

Or with uv:

```bash
uv add pipecat-protoface
```

Install the Pipecat extras required by your application separately, such as the
transport, LLM, STT, or TTS services in your pipeline.

For this repo's WebRTC example:

```bash
uv sync --extra dev --extra example
```

If you prefer pip:

```bash
pip install -e ".[dev,example]"
```

## Prerequisites

- Protoface API key (`PROTOFACE_API_KEY`)
- Protoface avatar ID (`PROTOFACE_AVATAR_ID`)
- API keys for the Pipecat services used by your pipeline

## Usage

`ProtofaceVideoService` starts a hosted Protoface avatar session. It consumes
`TTSAudioRawFrame` audio from a Pipecat pipeline and emits synchronized
`SpeechOutputAudioRawFrame` and `OutputImageRawFrame` frames downstream.

Your application still chooses the Pipecat transport for the end user.

```python
import os

from pipecat_protoface import ProtofaceVideoService

protoface = ProtofaceVideoService(
    api_key=os.environ["PROTOFACE_API_KEY"],
    avatar_id=os.environ["PROTOFACE_AVATAR_ID"],
)

pipeline = Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    llm,
    tts,
    protoface,  # Video avatar (returns synchronized audio/video)
    transport.output(),
    context_aggregator.assistant(),
])
```

For realtime speech-to-speech services, place `ProtofaceVideoService` after your
realtime LLM service and before the output transport:

```python
pipeline = Pipeline([
    transport.input(),
    gemini_live,
    protoface,
    transport.output(),
])
```

See [examples/video_service.py](examples/video_service.py) for a complete WebRTC
example using Gemini Live.

## Session Startup

`ProtofaceVideoService` starts a hosted Protoface avatar session when the
pipeline starts. Audio produced before the avatar session is ready is buffered,
then streamed once the connection is available.

## Running the Example

1. Install dependencies:

```bash
uv sync --extra dev --extra example
```

2. Set up your environment:

```bash
cp env.example .env
# Edit .env with your keys:
# PROTOFACE_API_KEY
# PROTOFACE_AVATAR_ID
# GOOGLE_API_KEY or GEMINI_API_KEY
#
# GEMINI_LIVE_MODEL and GEMINI_VOICE are optional.
```

3. Run the `ProtofaceVideoService` example with the built-in WebRTC transport:

```bash
uv run python examples/video_service.py -t webrtc
```

The example reads either `GOOGLE_API_KEY` or `GEMINI_API_KEY`.
By default, the plugin connects to `https://api.protoface.com`; set
`PROTOFACE_API_URL` only when testing against another Protoface environment.
Set `PROTOFACE_DEBUG_MEDIA=1` to log avatar audio/video frame counts during
local testing.

## Compatibility

- Pipecat v1.4.0+
- Python 3.11+
- Built-in WebRTC transport for the included example

## License

Apache-2.0 - see [LICENSE](LICENSE)

## Support

- [Protoface documentation](https://docs.protoface.com)
- [Pipecat Discord](https://discord.gg/pipecat) (`#community-integrations`)
