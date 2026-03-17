from ingest.chunker import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP


def test_empty_string():
    assert chunk_text("") == []


def test_short_text_single_chunk():
    text = "Hello world"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_size():
    text = "a" * 1200
    chunks = chunk_text(text)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_overlap():
    text = "a" * (CHUNK_SIZE + CHUNK_OVERLAP + 10)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    # second chunk should start CHUNK_SIZE - CHUNK_OVERLAP chars in
    assert chunks[1] == text[CHUNK_SIZE - CHUNK_OVERLAP: 2 * CHUNK_SIZE - CHUNK_OVERLAP]
