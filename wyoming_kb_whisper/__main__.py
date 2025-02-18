import argparse
import asyncio
import logging
from functools import partial

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import KBWhisperEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Name of kb-whisper model to use (large, medium, small, base, tiny)",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Data directory to check for downloaded models",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="kb-whisper",
                description="KB Whisper transcription",
                attribution=Attribution(
                    name="Alex Gustafsson",
                    url="http://github.com/alexgustafsson",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="kb-whisper-" + args.model,
                        description="kb-whisper-" + args.model,
                        attribution=Attribution(
                            name="KBLab",
                            url="https://huggingface.co/KBLab",
                        ),
                        installed=True,
                        languages=["sv"],
                        version=__version__,
                    )
                ],
            )
        ],
    )

    # Load model
    _LOGGER.info("Loading %s", args.model)
    torch_dtype = torch.float32
    model_id = "KBLab/kb-whisper-" + args.model

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True, cache_dir=args.data_dir
    )
    model.to("cpu")
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device="cpu",
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    pipe_lock = asyncio.Lock()
    await server.run(
        partial(
            KBWhisperEventHandler,
            wyoming_info,
            pipe,
            pipe_lock,
        )
    )


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
