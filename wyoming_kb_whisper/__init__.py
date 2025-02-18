"""Wyoming server for faster-whisper."""

from pathlib import Path

_DIR = Path(__file__).parent
_VERSION_PATH = _DIR / "VERSION"

__version__ = "0.1.0"

__all__ = ["__version__"]
