import pytest
from src.processing.chunking import split_text

def test_split_text_empty():
    assert split_text("") == []
    assert split_text("   ") == []

def test_split_text_single_short_sentence():
    text = "This is a short sentence."
    chunks = split_text(text, max_words=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_split_text_multiple_sentences():
    text = "Hello world. This is a test. We should split this well."
    chunks = split_text(text, max_words=6, overlap=2)
    assert len(chunks) >= 2
    # Ensure all words are present across chunks roughly

def test_split_text_long_sentence_fallback():
    # Sentence without punctuation longer than max_words
    text = "this is a very long sentence that has no punctuation and just keeps going on and on for many words without stopping so we need to fallback on word splitting"
    chunks = split_text(text, max_words=10, overlap=2)
    assert sum(len(c.split()) for c in chunks) >= len(text.split())
    for chunk in chunks:
        assert len(chunk.split()) <= 10
