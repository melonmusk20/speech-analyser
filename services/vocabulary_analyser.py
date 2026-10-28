"""Vocabulary analysis (FR-06)."""
from collections import Counter
from functools import lru_cache

from utils.text_utils import STOP_WORDS, letter_count

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
MATTR_WINDOW = 50

TTR_NOTE = (
    "Type-token ratio falls as a transcript gets longer, so compare it only between "
    "recordings of similar length. It is not a measure of language proficiency."
)


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    except Exception:
        return None


def _content_flags(tokens: list[str]) -> tuple[list[bool], str]:
    """Per-token content-word flag, using spaCy POS tags when available."""
    nlp = _load_spacy()
    if nlp is not None and tokens:
        try:
            from spacy.tokens import Doc

            doc = Doc(nlp.vocab, words=tokens)
            for _, proc in nlp.pipeline:
                doc = proc(doc)
            return [t.pos_ in CONTENT_POS for t in doc], "spaCy part-of-speech tagging"
        except Exception:
            pass
    return [t not in STOP_WORDS for t in tokens], "stop-word heuristic (spaCy model not installed)"


def moving_average_ttr(tokens: list[str], window: int = MATTR_WINDOW) -> float | None:
    """Length-independent diversity: mean TTR (%) over sliding windows."""
    if len(tokens) < window:
        return None
    counts = Counter(tokens[:window])
    distinct = len(counts)
    distinct_sum = distinct
    for i in range(window, len(tokens)):
        out, inn = tokens[i - window], tokens[i]
        counts[out] -= 1
        if counts[out] == 0:
            distinct -= 1
        if counts[inn] == 0:
            distinct += 1
        counts[inn] += 1
        distinct_sum += distinct
    n_windows = len(tokens) - window + 1
    return 100 * distinct_sum / (n_windows * window)


def analyse_vocabulary(tokens: list[str], top_n: int = 15) -> dict:
    total = len(tokens)
    if not total:
        return {
            "total_words": 0, "unique_words": 0, "ttr": 0.0, "mattr": None, "avg_word_length": 0.0,
            "content_word_pct": 0.0, "top_content_words": [], "uncommon_words": [],
            "method": "n/a", "note": TTR_NOTE,
        }
    unique = len(set(tokens))
    flags, method = _content_flags(tokens)
    content = [t for t, f in zip(tokens, flags) if f]
    counts = Counter(tokens)
    return {
        "total_words": total,
        "unique_words": unique,
        "ttr": round(100 * unique / total, 2),
        "mattr": None if (m := moving_average_ttr(tokens)) is None else round(m, 2),
        "avg_word_length": round(sum(letter_count(t) for t in tokens) / total, 2),
        "content_word_pct": round(100 * len(content) / total, 2),
        "top_content_words": [{"word": w, "count": c} for w, c in Counter(content).most_common(top_n)],
        # Heuristic: long, non-stop words said once. Not a validated rarity measure.
        "uncommon_words": sorted({t for t in content if counts[t] == 1 and letter_count(t) >= 9})[:50],
        "method": method,
        "note": TTR_NOTE,
    }
