"""Tests for converting Azure formula results into serializable metadata."""

import unittest
from types import SimpleNamespace

from src.data_pipeline.formula_parser import (
    build_formula_chunks,
    extract_formulas_from_pages,
)


class FormulaParserTests(unittest.TestCase):
    def test_extracts_latex_page_kind_and_polygon(self):
        pages = [
            SimpleNamespace(
                page_number=2,
                formulas=[
                    SimpleNamespace(
                        kind="display",
                        value=r"y = ax + b",
                        confidence=0.99,
                        polygon=[1.0, 2.0, 3.0, 4.0],
                    )
                ],
            )
        ]

        formulas = extract_formulas_from_pages(pages)

        self.assertEqual(len(formulas), 1)
        self.assertEqual(formulas[0]["formula_id"], "page_2_formula_1")
        self.assertEqual(formulas[0]["page_number"], 2)
        self.assertEqual(formulas[0]["kind"], "display")
        self.assertEqual(formulas[0]["latex"], r"y = ax + b")
        self.assertEqual(formulas[0]["polygon"], [1.0, 2.0, 3.0, 4.0])

    def test_handles_pages_without_formulas(self):
        pages = [SimpleNamespace(page_number=1, formulas=None)]
        self.assertEqual(extract_formulas_from_pages(pages), [])

    def test_handles_missing_pages(self):
        self.assertEqual(extract_formulas_from_pages(None), [])

    def test_builds_searchable_formula_chunks(self):
        formulas = [
            {
                "formula_id": "page_3_formula_1",
                "page_number": 3,
                "kind": "inline",
                "latex": r"E = mc^2",
                "confidence": 0.98,
                "polygon": [],
            }
        ]

        chunks = build_formula_chunks(
            formulas,
            {"source_file": "Physics Paper.pdf"},
            start_index=7,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "Physics_Paper_pdf_page_3_formula_1")
        self.assertIn(r"E = mc^2", chunks[0]["content"])
        self.assertEqual(chunks[0]["metadata"]["chunk_index"], 7)
        self.assertEqual(chunks[0]["metadata"]["content_type"], "formula")
        self.assertEqual(chunks[0]["metadata"]["page_number"], 3)


if __name__ == "__main__":
    unittest.main()
