"""SeekDB collection backend for OpenViking."""

from .adapter import SeekDBCollectionAdapter
from .collection import SeekDBCollection

__all__ = ["SeekDBCollection", "SeekDBCollectionAdapter"]
__version__ = "0.1.0"
