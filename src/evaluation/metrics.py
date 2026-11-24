"""Standard information-retrieval metrics used by the RAG evaluation runner."""

from typing import Iterable, List, Set


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")


def _unique_relevant(relevant_ids: Iterable[str]) -> Set[str]:
    return {item for item in relevant_ids if item}


def precision_at_k(
    retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Return the fraction of the first ``k`` results that are relevant."""
    _validate_k(k)
    retrieved: List[str] = list(retrieved_ids)[:k]
    relevant = _unique_relevant(relevant_ids)
    hits = sum(1 for item in retrieved if item in relevant)
    return hits / k


def recall_at_k(
    retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Return the fraction of all relevant results found in the first ``k``."""
    _validate_k(k)
    relevant = _unique_relevant(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(list(retrieved_ids)[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int = None
) -> float:
    """Return the reciprocal rank of the first relevant result, or zero."""
    relevant = _unique_relevant(relevant_ids)
    retrieved = list(retrieved_ids)
    if k is not None:
        _validate_k(k)
        retrieved = retrieved[:k]
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
