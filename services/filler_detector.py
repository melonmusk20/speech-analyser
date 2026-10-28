"""Filler word detection (FR-05).

Results are estimates: speech recognisers often drop hesitation sounds, and words
like "like" or "so" are only sometimes fillers.
"""
from utils.text_utils import tokenize

# "definite" fillers are always counted; "context" words are counted only when
# punctuation suggests a pause around them (e.g. "So, ..." / "..., like, ...") or
# when the user includes them in manual review.
DEFAULT_DICTIONARY = {
    "definite": ["um", "uh", "uhm", "erm", "er", "hmm", "you know", "i mean", "sort of", "kind of"],
    "context": ["basically", "actually", "literally", "well", "like", "so", "okay", "right"],
}

LIMITATIONS = (
    "Filler counts are estimates. Speech recognition may omit hesitation sounds, and "
    "context-dependent words (like, so, right...) are only counted when punctuation suggests "
    "a filler use. Use the review table to correct them."
)


def _entries(words: list[str]) -> list[tuple[str, ...]]:
    return [tuple(tokenize(w)) for w in words if tokenize(w)]


def _is_likely_filler(text: str, start: int, end: int) -> bool:
    before = text[:start].rstrip()[-1:]
    after = text[end:].lstrip()[:1]
    return before == "," or after == ","


def detect_fillers(
    spans: list[tuple[str, int, int]],
    text: str,
    dictionary: dict | None = None,
    token_times: list[tuple[float, float]] | None = None,
    overrides: dict[int, bool] | None = None,
) -> list[dict]:
    """Find non-overlapping dictionary matches, preferring the longest phrase.

    `overrides` maps a match's start token index to True/False (manual review).
    """
    dictionary = dictionary or DEFAULT_DICTIONARY
    overrides = overrides or {}
    table: dict[tuple[str, ...], str] = {}
    for tier in ("context", "definite"):  # definite wins if a phrase is in both
        for entry in _entries(dictionary.get(tier, [])):
            table[entry] = tier
    if not table:
        return []
    max_len = max(len(e) for e in table)
    tokens = [s[0] for s in spans]

    matches, i = [], 0
    while i < len(tokens):
        hit = None
        for n in range(min(max_len, len(tokens) - i), 0, -1):
            key = tuple(tokens[i : i + n])
            if key in table:
                hit = (key, n)
                break
        if not hit:
            i += 1
            continue
        key, n = hit
        tier = table[key]
        likely = tier == "definite" or _is_likely_filler(text, spans[i][1], spans[i + n - 1][2])
        matches.append(
            {
                "index": i,
                "length": n,
                "phrase": " ".join(key),
                "type": tier,
                "likely": likely,
                "included": overrides.get(i, likely),
                "time": token_times[i][0] if token_times else None,
                "context": " ".join(tokens[max(0, i - 4) : i + n + 4]),
            }
        )
        i += n
    return matches


def summarize_fillers(matches: list[dict], total_words: int) -> dict:
    included = [m for m in matches if m["included"]]
    counts: dict[str, int] = {}
    for m in included:
        counts[m["phrase"]] = counts.get(m["phrase"], 0) + 1
    return {
        "total": len(included),
        "per_100_words": round(100 * len(included) / total_words, 2) if total_words else 0.0,
        "by_phrase": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "definite": sum(1 for m in included if m["type"] == "definite"),
        "context_included": sum(1 for m in included if m["type"] == "context"),
        "context_excluded": sum(1 for m in matches if m["type"] == "context" and not m["included"]),
        "limitations": LIMITATIONS,
    }
