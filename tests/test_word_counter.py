from services.word_counter import search_word, text_stats, word_frequencies
from utils.text_utils import count_words, tokenize


def test_contractions_and_hyphens_are_one_word():
    assert tokenize("I'm a well-known don't") == ["i'm", "a", "well-known", "don't"]


def test_punctuation_alone_is_not_a_word():
    assert count_words("Hello -- , ... !") == 1
    assert count_words("") == 0


def test_spec_example():
    # I'm, learning, Python, and, it's, useful -> 6 tokens
    assert count_words("I'm learning Python, and it's useful!") == 6


def test_case_normalised_frequency():
    tokens = tokenize("Python python PYTHON code")
    top = word_frequencies(tokens, 10)
    assert top[0] == {"word": "python", "count": 3, "percentage": 75.0}


def test_stop_word_filter_changes_ranking_not_tokens():
    tokens = tokenize("the cat and the dog and the bird")
    assert word_frequencies(tokens, 1)[0]["word"] == "the"
    assert word_frequencies(tokens, 1, exclude_stop_words=True)[0]["count"] == 1
    assert len(tokens) == 8


def test_search_and_empty():
    assert search_word(tokenize("a b a"), "A")["count"] == 2
    assert word_frequencies([], 10) == []


def test_text_stats():
    s = text_stats("One two. Three!\n\nFour", tokenize("One two. Three!\n\nFour"), 2)
    assert (s["words"], s["sentences"], s["paragraphs"], s["segments"]) == (4, 3, 2, 2)
