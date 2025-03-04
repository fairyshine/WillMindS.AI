"""Python Package named WillMindS"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .monitor import Monitor

__all__ = [
    "__version__",
    "config",
    "logger"
]

# Official PEP 396
try:
    __version__ = version("willminds")
except PackageNotFoundError:
    __version__ = "unknown version"

monitor = Monitor()
config = monitor.config
logger = monitor.logger

