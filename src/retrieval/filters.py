"""Pure helpers for building Azure AI Search filter expressions."""

from typing import Optional


def build_source_file_filter(source_file: Optional[str]) -> Optional[str]:
    """Return an OData filter for a source filename, or ``None`` for all files."""
    if not source_file:
        return None

    # OData string literals escape a single quote by doubling it.
    safe_source_file = source_file.replace("'", "''")
    return f"source_file eq '{safe_source_file}'"
