"""Ingestion error types used to decide whether a job is worth retrying."""


class PermanentIngestionError(Exception):
    """The document can never be ingested as-is (unsupported, corrupt, no text).

    The worker marks the document 'failed' immediately instead of retrying.
    Any other exception is treated as transient and retried.
    """
