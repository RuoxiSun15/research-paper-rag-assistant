from src.data_pipeline.text_chunker import TextChunker


def test_splitter_respects_configured_size():
    chunker = TextChunker(chunk_size=40, chunk_overlap_ratio=0.25)
    text = "First paragraph has useful content. Second sentence follows. " * 3
    chunks = chunker._split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_splitter_prefers_paragraph_boundaries():
    chunker = TextChunker(chunk_size=30, chunk_overlap_ratio=0)
    chunks = chunker._split_text("Alpha section.\n\nBeta section with more words.")

    assert chunks[0] == "Alpha section."


def test_empty_text_produces_no_chunks():
    chunker = TextChunker(chunk_size=20, chunk_overlap_ratio=0.2)
    assert chunker.chunk_text("   ") == []
