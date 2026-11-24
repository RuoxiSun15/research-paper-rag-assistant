"""Tests for vision analysis formatting and local image encoding."""

from pathlib import Path

from src.data_pipeline.vision_analyzer import VisionAnalyzer, format_vision_analysis


def test_formats_structured_visual_analysis():
    text = format_vision_analysis({
        "figure_type": "architecture diagram",
        "summary": "Three embedding models are compared.",
        "key_elements": ["head entity", "relation", "tail entity"],
        "relationships": ["inputs flow toward a scoring function"],
        "limitations": [],
    })

    assert "Figure type: architecture diagram" in text
    assert "head entity; relation; tail entity" in text
    assert "inputs flow toward a scoring function" in text


def test_encodes_png_as_data_url(tmp_path: Path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    encoded = VisionAnalyzer._image_data_url(str(image))

    assert encoded.startswith("data:image/png;base64,")
