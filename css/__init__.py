from pathlib import Path

from css._version import __version__

SRC_ROOT = Path(__file__).resolve().parent

__all__ = ["SRC_ROOT", "__version__"]
