"""Tests for optional source-file filtering in Azure Search retrieval."""

import unittest

from src.retrieval.filters import build_source_file_filter


class SourceFileFilterTests(unittest.TestCase):
    def test_builds_filter_for_source_file(self):
        self.assertEqual(
            build_source_file_filter("TimeGRU.pdf"),
            "source_file eq 'TimeGRU.pdf'",
        )

    def test_retrieve_escapes_single_quotes_in_file_name(self):
        self.assertEqual(
            build_source_file_filter("Author's paper.pdf"),
            "source_file eq 'Author''s paper.pdf'",
        )

    def test_without_source_file_keeps_full_corpus_search(self):
        self.assertIsNone(build_source_file_filter(None))
        self.assertIsNone(build_source_file_filter(""))


if __name__ == "__main__":
    unittest.main()
