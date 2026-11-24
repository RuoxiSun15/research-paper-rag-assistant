"""Tests for serializing figures and building searchable chunks."""

import struct
from types import SimpleNamespace

from src.data_pipeline.figure_parser import build_figure_chunks, extract_figures


def test_extracts_figure_metadata():
    figure = SimpleNamespace(
        id="1.1",
        caption=SimpleNamespace(content="Accuracy by model", elements=["/p/1"]),
        bounding_regions=[SimpleNamespace(page_number=2, polygon=[1, 2, 3, 4])],
        elements=["/figures/1"],
    )

    figures = extract_figures([figure])

    assert figures[0]["figure_id"] == "1.1"
    assert figures[0]["caption"] == "Accuracy by model"
    assert figures[0]["page_numbers"] == [2]
    assert figures[0]["bounding_regions"][0]["polygon"] == [1, 2, 3, 4]


def test_builds_searchable_figure_chunk():
    chunks = build_figure_chunks(
        [{
            "figure_id": "2.1",
            "caption": "Embedding architecture",
            "page_numbers": [4],
            "image_path": "/tmp/figures/2.1.png",
            "vision_analysis": {
                "figure_type": "architecture diagram",
                "summary": "Inputs flow into a scoring layer.",
                "key_elements": ["head", "relation", "tail"],
            },
        }],
        {"source_file": "paper.pdf"},
        start_index=3,
    )

    assert chunks[0]["metadata"]["content_type"] == "figure"
    assert chunks[0]["metadata"]["page_number"] == 4
    assert "Embedding architecture" in chunks[0]["content"]
    assert "Visual summary: Inputs flow into a scoring layer." in chunks[0]["content"]
    assert chunks[0]["metadata"]["figure_type"] == "architecture diagram"
    assert chunks[0]["chunk_id"] == "paper_pdf_2_1"


def test_skips_tiny_decorative_png(tmp_path):
    image_path = tmp_path / "publisher-button.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 143, 32)
    )

    chunks = build_figure_chunks(
        [{
            "figure_id": "1.1",
            "caption": "",
            "page_numbers": [1],
            "image_path": str(image_path),
        }],
        {"source_file": "paper.pdf"},
    )

    assert chunks == []
