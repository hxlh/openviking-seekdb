"""Plugin-specific exceptions."""


class SeekDBError(RuntimeError):
    """Base error for the SeekDB adapter."""


class SeekDBConfigurationError(SeekDBError, ValueError):
    """The OpenViking or SeekDB configuration is invalid."""


class SeekDBConnectionError(SeekDBError):
    """SeekDB could not be reached or the connection was lost."""


class SeekDBCollectionMissingError(SeekDBError):
    """Collection catalog metadata and physical storage disagree."""


class SeekDBUnsupportedError(SeekDBError, NotImplementedError):
    """The requested operation is not supported by SeekDB."""
