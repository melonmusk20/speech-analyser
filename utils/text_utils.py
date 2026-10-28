"""Shared tokenisation and text helpers. Every module counts words through here."""
import html
import re

# A word is a run of letters/digits, optionally joined by apostrophes or hyphens
# ("don't", "well-known"). Punctuation alone is never a word.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

STOP_WORDS = frozenset(
    """a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have haven't
    having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor
    not of off on once only or other ought our ours ourselves out over own same shan't she she'd
    she'll she's should shouldn't so some such than that that's the their theirs them themselves
    then there there's these they they'd they'll they're they've this those through to too under
    until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
    where's which while who who's whom why why's will with won't would wouldn't you you'd you'll
    you're you've your yours yourself yourselves also just yeah""".split()
)


def normalize_token(token: str) -> str:
    return token.lower().replace("’", "'")


def tokenize_spans(text: str) -> list[tuple[str, int, int]]:
    """Return (normalised_token, char_start, char_end) for every word in text."""
    return [(normalize_token(m.group()), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def tokenize(text: str) -> list[str]:
    return [t for t, _, _ in tokenize_spans(text)]


def count_words(text: str) -> int:
    return len(tokenize(text))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if _TOKEN_RE.search(s)]


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if _TOKEN_RE.search(p)]


def letter_count(token: str) -> int:
    """Letters/digits in a token (apostrophes and hyphens are not counted)."""
    return sum(c.isalnum() for c in token)


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


def highlight_html(text: str, spans: list[tuple[str, int, int]], ranges: list[tuple[int, int]]) -> str:
    """Escape text as HTML, wrapping token ranges [start_idx, end_idx) in <mark>."""
    char_ranges = []
    for s, e in sorted(ranges):
        if not (0 <= s < e <= len(spans)):
            continue
        cs, ce = spans[s][1], spans[e - 1][2]
        if char_ranges and cs <= char_ranges[-1][1]:
            char_ranges[-1] = (char_ranges[-1][0], max(ce, char_ranges[-1][1]))
        else:
            char_ranges.append((cs, ce))
    out, pos = [], 0
    for cs, ce in char_ranges:
        out.append(html.escape(text[pos:cs]))
        out.append(f"<mark>{html.escape(text[cs:ce])}</mark>")
        pos = ce
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")
