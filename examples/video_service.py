"""Example: Pipecat assistant with a Protoface avatar video service.

Run with Pipecat's built-in WebRTC transport:

    uv run --extra example python examples/video_service.py -t webrtc
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiVADParams,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner
from pipecat_protoface import ProtofaceVideoService

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=16_000,
        audio_in_channels=1,
        audio_out_enabled=True,
        audio_out_sample_rate=16_000,
        audio_out_channels=1,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=_env_int("PROTOFACE_VIDEO_WIDTH", 512),
        video_out_height=_env_int("PROTOFACE_VIDEO_HEIGHT", 512),
        video_out_framerate=_env_int("PROTOFACE_VIDEO_FPS", 25),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY.")

    llm_kwargs: dict[str, Any] = {
        "api_key": google_api_key,
        "system_instruction": os.getenv(
            "GEMINI_SYSTEM_INSTRUCTION",
            "You are a concise realtime voice assistant. Keep responses short.",
        ),
    }
    llm_settings_kwargs: dict[str, Any] = {}
    if model := os.getenv("GEMINI_LIVE_MODEL"):
        llm_settings_kwargs["model"] = model
    if voice := os.getenv("GEMINI_VOICE"):
        llm_settings_kwargs["voice"] = voice
    local_turns = _env_bool("GEMINI_LOCAL_TURNS", default=True)
    if local_turns:
        llm_settings_kwargs["vad"] = GeminiVADParams(disabled=True)
    if llm_settings_kwargs:
        llm_kwargs["settings"] = GeminiLiveLLMService.Settings(**llm_settings_kwargs)

    llm = GeminiLiveLLMService(**llm_kwargs)
    logger.info(
        "Gemini Live configured: model={} voice={} local_turns={}",
        llm_settings_kwargs.get("model", "pipecat-default"),
        llm_settings_kwargs.get("voice", "pipecat-default"),
        local_turns,
    )
    avatar = ProtofaceVideoService(
        api_key=_required_env("PROTOFACE_API_KEY"),
        avatar_id=_required_env("PROTOFACE_AVATAR_ID"),
        api_url=os.getenv("PROTOFACE_API_URL"),
        metadata={
            "source": "pipecat-video-service-example",
            "pipecat_session_id": runner_args.session_id,
        },
    )

    context = LLMContext()
    user_aggregator_params = None
    if local_turns:
        user_aggregator_params = LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16_000,
                params=VADParams(
                    confidence=_env_float("PIPECAT_VAD_CONFIDENCE", 0.7),
                    start_secs=_env_float("PIPECAT_VAD_START_SECS", 0.2),
                    stop_secs=_env_float("PIPECAT_VAD_STOP_SECS", 0.2),
                    min_volume=_env_float("PIPECAT_VAD_MIN_VOLUME", 0.2),
                ),
            ),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=_env_float("PIPECAT_USER_SPEECH_TIMEOUT", 0.2),
                        wait_for_transcript=False,
                    )
                ]
            ),
        )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=user_aggregator_params,
        realtime_service_mode=True,
    )

    processors = [
        transport.input(),
        context_aggregator.user(),
        llm,
        context_aggregator.assistant(),
        avatar,
        transport.output(),
    ]
    pipeline = Pipeline(processors)
    worker = PipelineWorker(
        pipeline,
        enable_rtvi=False,
        enable_turn_tracking=False,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        params=PipelineParams(
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=16_000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    runner = WorkerRunner(
        handle_sigint=runner_args.handle_sigint,
        handle_sigterm=runner_args.handle_sigterm,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport: Any, _client: Any) -> None:  # pyright: ignore[reportUnusedFunction]
        logger.info("Client connected")
        await context_aggregator.user().queue_frame(LLMRunFrame(), FrameDirection.DOWNSTREAM)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport: Any, _client: Any) -> None:  # pyright: ignore[reportUnusedFunction]
        logger.info("Client disconnected")
        await runner.cancel(reason="client disconnected")

    await runner.run(worker)


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name}.")
    return value


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float.") from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
