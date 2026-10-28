"""Word counts and word-frequency analysis (FR-02, FR-04)."""
from collections import Counter

from utils.text_utils import STOP_WORDS, split_paragraphs, split_sentences


def text_stats(text: str, tokens: list[str], segment_count: int = 0) -> dict:
    return {
        "words": len(tokens),
        "unique_words": len(set(tokens)),
        "sentences": len(split_sentences(text)),
        "paragraphs": len(split_paragraphs(text)),
        "segments": segment_count,
    }


def word_frequencies(tokens: list[str], top_n: int = 25, exclude_stop_words: bool = False) -> list[dict]:
    """Top words with count and percentage of all counted words.

    When stop words are excluded, percentages are relative to the remaining words.
    """
    counted = [t for t in tokens if t not in STOP_WORDS] if exclude_stop_words else tokens
    total = len(counted)
    if not total:
        return []
    return [
        {"word": w, "count": c, "percentage": round(100 * c / total, 2)}
        for w, c in Counter(counted).most_common(top_n)
    ]


def search_word(tokens: list[str], query: str) -> dict:
    q = query.strip().lower().replace("’", "'")
    count = sum(1 for t in tokens if t == q) if q else 0
    return {
        "word": q,
        "count": count,
        "percentage": round(100 * count / len(tokens), 2) if tokens else 0.0,
    }
