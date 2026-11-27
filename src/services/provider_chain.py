"""Shared 'provider:model' fallback-chain parsing for ModelService/EmbeddingService."""


def parse_provider_chain(raw: str, setting_name: str) -> list[tuple[str, str]]:
    """Parse a comma-separated ``provider:model`` list, e.g.

        "openai:gpt-4o-mini,openai:gpt-4o"

    into ``[("openai", "gpt-4o-mini"), ("openai", "gpt-4o")]``.
    The first entry is the primary model; the rest are tried in order as
    fallbacks if an earlier entry errors or is rate-limited.

    *setting_name* is the environment variable name, used in error messages.
    """
    chain = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        provider, sep, model = entry.partition(":")
        if not sep or not provider.strip() or not model.strip():
            raise RuntimeError(
                f"{setting_name} entries must look like 'provider:model' (got {entry!r})"
            )
        chain.append((provider.strip().lower(), model.strip()))
    if not chain:
        raise RuntimeError(f"{setting_name} must define at least one 'provider:model' entry")
    return chain
