"""Event handler for clients of the server."""

import asyncio
import logging
import os
import tempfile
import wave
from typing import Optional

from transformers import pipeline

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class KBWhisperEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        pipe: pipeline,
        pipe_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.pipe = pipe
        self.pipe_lock = pipe_lock
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped. Transcribing")
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            async with self.pipe_lock:
                text = self.pipe(
                    self._wav_path,
                    chunk_length_s=30,
                    generate_kwargs={"task": "transcribe", "language": "sv"},
                )["text"]

            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            return False

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
