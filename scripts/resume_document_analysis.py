#!/usr/bin/env python3
"""Resume a completed Azure Document Intelligence result without re-uploading it."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config
from src.data_pipeline.embedder import Embedder
from src.data_pipeline.figure_parser import build_figure_chunks
from src.data_pipeline.formula_parser import build_formula_chunks
from src.data_pipeline.indexer import AzureSearchIndexer
from src.data_pipeline.text_chunker import TextChunker
from src.data_pipeline.vision_analyzer import (
    VisionAnalyzer,
    is_meaningful_figure_image,
)
from src.utils.azure_cli_credential import AzureCliCredentialLite


def load_analysis(result_id: str) -> dict:
    """Download the JSON for an already completed layout analysis."""
    config = get_config()
    url = (
        f"{config.doc_intelligence_endpoint.rstrip('/')}"
        f"/documentintelligence/documentModels/prebuilt-layout/analyzeResults/"
        f"{result_id}?api-version=2024-11-30"
    )
    token = AzureCliCredentialLite().get_token(
        "https://cognitiveservices.azure.com/.default"
    )
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token.token}"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))["analyzeResult"]


def parse_formulas(result: dict) -> list[dict]:
    formulas = []
    for page in result.get("pages", []):
        page_number = page.get("pageNumber")
        for index, formula in enumerate(page.get("formulas", []), start=1):
            formulas.append({
                "formula_id": f"page_{page_number}_formula_{index}",
                "page_number": page_number,
                "kind": formula.get("kind"),
                "latex": formula.get("value", ""),
                "confidence": formula.get("confidence"),
                "polygon": formula.get("polygon", []),
            })
    return formulas


def parse_figures(result: dict, figure_dir: Path) -> list[dict]:
    figures = []
    for index, figure in enumerate(result.get("figures", []), start=1):
        figure_id = figure.get("id") or f"figure_{index}"
        regions = figure.get("boundingRegions", [])
        image_path = figure_dir / f"{figure_id}.png"
        figures.append({
            "figure_id": figure_id,
            "caption": (figure.get("caption") or {}).get("content", ""),
            "caption_elements": (figure.get("caption") or {}).get("elements", []),
            "page_numbers": [
                region.get("pageNumber") for region in regions
                if region.get("pageNumber") is not None
            ],
            "bounding_regions": regions,
            "elements": figure.get("elements", []),
            "image_path": str(image_path.absolute()) if image_path.exists() else None,
        })
    return figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_id")
    parser.add_argument("pdf_path")
    parser.add_argument("--max-text-chunks", type=int)
    parser.add_argument(
        "--text-chunk-indices",
        help="Comma-separated text chunk indices to embed (for low-quota tests)",
    )
    parser.add_argument("--max-formula-chunks", type=int)
    parser.add_argument("--max-figure-chunks", type=int)
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--embedding-batch-delay", type=float, default=0.1)
    parser.add_argument("--replace-source", action="store_true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    result = load_analysis(args.result_id)
    metadata = {
        "source_file": pdf_path.name,
        "file_path": str(pdf_path.absolute()),
        "page_count": len(result.get("pages", [])),
        "file_size_bytes": pdf_path.stat().st_size,
    }
    formulas = parse_formulas(result)
    figures = parse_figures(
        result,
        Path(get_config().figure_output_dir) / pdf_path.stem,
    )
    metadata.update({"formula_count": len(formulas), "figure_count": len(figures)})
    print(
        f"loaded pages={metadata['page_count']} formulas={len(formulas)} "
        f"figures={len(figures)}",
        flush=True,
    )

    figures = [
        figure for figure in figures
        if figure.get("image_path")
        and is_meaningful_figure_image(figure["image_path"])
    ]
    if args.max_figure_chunks is not None:
        figures = figures[:args.max_figure_chunks]
    if get_config().vision_analysis_enabled:
        figures = VisionAnalyzer().analyze_figures(figures)
    analyzed_count = sum(bool(item.get("vision_analysis")) for item in figures)
    print(f"vision_analyzed={analyzed_count}", flush=True)

    text_chunks = TextChunker().chunk_text(result.get("content", ""), metadata)
    formula_chunks = build_formula_chunks(formulas, metadata, len(text_chunks))
    if args.text_chunk_indices:
        selected_indices = {
            int(value.strip())
            for value in args.text_chunk_indices.split(",")
            if value.strip()
        }
        text_chunks = [
            chunk for chunk in text_chunks
            if chunk["metadata"]["chunk_index"] in selected_indices
        ]
    if args.max_text_chunks is not None:
        text_chunks = text_chunks[:args.max_text_chunks]
    if args.max_formula_chunks is not None:
        formula_chunks = formula_chunks[:args.max_formula_chunks]
    figure_chunks = build_figure_chunks(
        figures,
        metadata,
        len(text_chunks) + len(formula_chunks),
    )
    chunks = text_chunks + formula_chunks + figure_chunks
    print(
        f"chunks text={len(text_chunks)} formula={len(formula_chunks)} "
        f"figure={len(figure_chunks)} total={len(chunks)}",
        flush=True,
    )

    embedder = Embedder()
    embedder.MAX_BATCH_SIZE = args.embedding_batch_size
    embedder.BATCH_DELAY_SECONDS = args.embedding_batch_delay
    embedder.embed_chunks(chunks)
    print(f"embedded={len(chunks)}", flush=True)
    indexer = AzureSearchIndexer()
    indexer.create_index(recreate=False)
    if args.replace_source:
        deleted = indexer.delete_documents_by_source(pdf_path.name)
        print(f"deleted_existing={deleted}", flush=True)
    indexed = indexer.index_documents(chunks)
    print(f"indexed={json.dumps(indexed)}", flush=True)


if __name__ == "__main__":
    main()
