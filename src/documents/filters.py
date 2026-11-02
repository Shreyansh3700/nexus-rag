"""Helpers for building Milvus scalar filter expressions safely."""


def milvus_filter_value(value: str) -> str:
    """Escape a string for use inside a double-quoted Milvus filter literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def document_filter(document_id: str) -> str:
    """Filter expression that matches every chunk of one document."""
    return f'document_id == "{milvus_filter_value(document_id)}"'
