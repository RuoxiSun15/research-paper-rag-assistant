"""Convert Azure Document Intelligence formulas into serializable RAG content."""

import re
from typing import Any, Dict, Iterable, List


def extract_formulas_from_pages(pages: Iterable[Any]) -> List[Dict[str, Any]]:
    """Return serializable formula metadata from analyzed document pages."""
    formulas: List[Dict[str, Any]] = []

    for page in pages or []:
        page_number = getattr(page, "page_number", None)
        for formula_index, formula in enumerate(
            getattr(page, "formulas", None) or [],
            start=1,
        ):
            polygon = getattr(formula, "polygon", None) or []
            formulas.append(
                {
                    "formula_id": f"page_{page_number}_formula_{formula_index}",
                    "page_number": page_number,
                    "kind": getattr(formula, "kind", None),
                    "latex": getattr(formula, "value", "") or "",
                    "confidence": getattr(formula, "confidence", None),
                    "polygon": list(polygon),
                }
            )

    return formulas


def build_formula_chunks(
    formulas: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any],
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    """Create text chunks that make LaTeX formulas searchable by the RAG pipeline."""
    chunks: List[Dict[str, Any]] = []
    source_file = metadata.get("source_file", "unknown")
    safe_source = re.sub(r"[^A-Za-z0-9_\-=]", "_", source_file)

    for offset, formula in enumerate(formulas or []):
        chunk_index = start_index + offset
        kind = formula.get("kind") or "unknown"
        page_number = formula.get("page_number")
        latex = formula.get("latex", "")
        content = (
            f"Mathematical formula ({kind}) on page {page_number}.\n"
            f"LaTeX: {latex}"
        )

        chunk_metadata = metadata.copy()
        chunk_metadata.update(
            {
                "chunk_index": chunk_index,
                "chunk_size": len(content),
                "content_type": "formula",
                "page_number": page_number,
                "formula_id": formula.get("formula_id"),
                "formula_kind": kind,
                "formula_latex": latex,
                "formula_confidence": formula.get("confidence"),
            }
        )

        chunks.append(
            {
                "content": content,
                "metadata": chunk_metadata,
                "chunk_id": f"{safe_source}_{formula.get('formula_id', chunk_index)}",
            }
        )

    return chunks
