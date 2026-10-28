"""Repetition analysis (FR-07)."""
from collections import Counter

from utils.text_utils import STOP_WORDS

# Words often repeated on purpose for emphasis; still reported, but flagged.
EMPHASIS_WORDS = {"very", "really", "so", "no", "yes", "ha", "bye", "more", "many", "far"}


def _time(token_times, idx):
    return token_times[idx][0] if token_times else None


def find_consecutive_repeats(tokens, token_times=None, max_len=8) -> list[dict]:
    """Immediate repeats: "I I think", "I was going I was going"."""
    covered = [False] * len(tokens)
    found = []
    for n in range(max_len, 0, -1):
        i = 0
        while i + 2 * n <= len(tokens):
            unit = tokens[i : i + n]
            if not any(covered[i : i + 2 * n]) and unit == tokens[i + n : i + 2 * n]:
                end, repeats = i + 2 * n, 2
                while tokens[end : end + n] == unit:
                    end += n
                    repeats += 1
                for k in range(i, end):
                    covered[k] = True
                found.append(
                    {
                        "type": "repeated word" if n == 1 else "repeated phrase (consecutive)",
                        "phrase": " ".join(unit),
                        "repeats": repeats,
                        "start": i,
                        "end": end,
                        "time": _time(token_times, i),
                        "possible_emphasis": n == 1 and unit[0] in EMPHASIS_WORDS,
                    }
                )
                i = end
            else:
                i += 1
    return sorted(found, key=lambda f: f["start"])


def find_repeated_phrases(tokens, token_times=None, min_len=2, max_len=8, min_repeats=2) -> list[dict]:
    """Phrases appearing at least `min_repeats` times anywhere in the transcript."""
    kept: dict[tuple, list[int]] = {}
    for n in range(min_len, max_len + 1):
        positions: dict[tuple, list[int]] = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            if all(t in STOP_WORDS for t in gram):  # common phrases like "i am the"
                continue
            positions.setdefault(gram, []).append(i)
        for gram, pos in positions.items():
            non_overlapping, last_end = [], -1
            for p in pos:
                if p >= last_end:
                    non_overlapping.append(p)
                    last_end = p + n
            if len(non_overlapping) >= min_repeats:
                kept[gram] = non_overlapping

    # Drop a phrase when a longer kept phrase contains it with the same count.
    subsumed = set()
    for gram, pos in kept.items():
        for n in range(min_len, len(gram)):
            for k in range(len(gram) - n + 1):
                sub = gram[k : k + n]
                if sub in kept and len(kept[sub]) == len(pos):
                    subsumed.add(sub)
    results = [
        {
            "type": "repeated phrase",
            "phrase": " ".join(g),
            "words": len(g),
            "count": len(p),
            "positions": p,
            "times": [_time(token_times, i) for i in p],
        }
        for g, p in kept.items()
        if g not in subsumed
    ]
    return sorted(results, key=lambda r: (-r["count"] * r["words"], r["phrase"]))[:100]


def find_overused_words(tokens, min_count=5, min_pct=2.0) -> list[dict]:
    total = len(tokens)
    return [
        {"word": w, "count": c, "percentage": round(100 * c / total, 2)}
        for w, c in Counter(tokens).most_common()
        if w not in STOP_WORDS and len(w) > 3 and c >= min_count and 100 * c / total >= min_pct
    ]


def analyse_repetition(tokens, token_times=None, min_len=2, max_len=8, min_repeats=2,
                       overuse_min_count=5, overuse_min_pct=2.0) -> dict:
    if not tokens:
        return {"consecutive": [], "phrases": [], "overused": [], "highlight_ranges": [], "events": 0}
    consecutive = find_consecutive_repeats(tokens, token_times, max_len)
    phrases = find_repeated_phrases(tokens, token_times, min_len, max_len, min_repeats)
    ranges = [(c["start"], c["end"]) for c in consecutive]
    for ph in phrases:
        ranges += [(p, p + ph["words"]) for p in ph["positions"]]
    return {
        "consecutive": consecutive,
        "phrases": phrases,
        "overused": find_overused_words(tokens, overuse_min_count, overuse_min_pct),
        "highlight_ranges": ranges,
        "events": len(consecutive) + sum(p["count"] - 1 for p in phrases),
        "note": "Repetition can be intentional (emphasis, quotations, common phrases); review before treating it as a problem.",
    }
