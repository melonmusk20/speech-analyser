"""Speech-to-text with faster-whisper (FR-01). Imported lazily so the rest of the app works without it."""
from dataclasses import asdict, dataclass, field
from typing import Callable

from utils.text_utils import tokenize


class TranscriptionError(Exception):
    """A user-presentable transcription failure; the user can retry."""


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    duration: float
    language: str | None
    segments: list[Segment]
    language_probability: float | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def to_dict(self) -> dict:
        return asdict(self) | {"text": self.text}


def align_token_times(tokens: list[str], result: TranscriptionResult):
    """Map each token to (start, end). Returns (times, precision) or (None, None).

    Only valid when `tokens` came from the unedited transcript.
    """
    words = result.words
    if words:
        times = [(w.start, w.end) for w in words for _ in tokenize(w.text)]
        if len(times) == len(tokens) and [t for w in words for t in tokenize(w.text)] == tokens:
            return times, "word"
    times, seg_tokens = [], []
    for seg in result.segments:
        toks = tokenize(seg.text)
        n = len(toks)
        for j, t in enumerate(toks):
            a = seg.start + (seg.end - seg.start) * j / n
            times.append((a, seg.start + (seg.end - seg.start) * (j + 1) / n))
            seg_tokens.append(t)
    if times and seg_tokens == tokens:
        return times, "segment"
    return None, None


_models: dict = {}


def load_model(model_size: str):
    if model_size in _models:
        return _models[model_size]
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed. Run `pip install -r requirements.txt` and try again."
        ) from exc
    try:
        _models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as exc:
        raise TranscriptionError(
            f"The '{model_size}' model could not be loaded. The first run needs an internet "
            f"connection to download it. Check your connection and retry. ({exc})"
        ) from exc
    return _models[model_size]


def transcribe(path, model_size: str, language: str | None, duration: float,
               on_progress: Callable[[float], None] | None = None) -> TranscriptionResult:
    model = load_model(model_size)
    try:
        gen, info = model.transcribe(str(path), language=language, word_timestamps=True, vad_filter=False)
        segments = []
        for seg in gen:
            segments.append(
                Segment(
                    seg.start, seg.end, seg.text.strip(),
                    [Word(w.start, w.end, w.word.strip(), w.probability) for w in (seg.words or [])],
                )
            )
            if on_progress and duration > 0:
                on_progress(min(1.0, seg.end / duration))
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed. You can retry. ({exc})") from exc
    return TranscriptionResult(duration, info.language, segments, info.language_probability)
