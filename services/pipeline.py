"""Runs every analysis module over one transcript and returns a structured result."""
import config
from services import filler_detector, fluency_analyser, repetition_analyser, vocabulary_analyser, word_counter
from services.transcription import TranscriptionResult, align_token_times
from utils.text_utils import tokenize_spans, count_words


def analyse(
    text: str,
    result: TranscriptionResult | None,
    silences: list[tuple[float, float]] | None,
    edited: bool = False,
    settings: config.AnalysisSettings | None = None,
    filler_dictionary: dict | None = None,
    filler_overrides: dict[int, bool] | None = None,
) -> dict:
    settings = settings or config.AnalysisSettings()
    spans = tokenize_spans(text)
    tokens = [s[0] for s in spans]
    total = len(tokens)
    duration = result.duration if result else 0.0
    warnings = []

    token_times, precision = (None, None)
    if result and not edited:
        token_times, precision = align_token_times(tokens, result)
    if edited:
        warnings.append("Transcript was edited: word timestamps for fillers and repetitions are unavailable.")
    elif result and token_times is None:
        warnings.append("Word timestamps are missing or unreliable, so timestamps are not shown.")
    if precision == "segment":
        warnings.append("Only segment-level timestamps are available; times are approximate.")
    if total == 0:
        warnings.append("No speech was detected in the transcript.")

    segments = result.segments if result else []
    if result and result.words:
        intervals = [(w.start, w.end) for w in result.words]
    else:
        intervals = [(s.start, s.end) for s in segments]

    matches = filler_detector.detect_fillers(spans, text, filler_dictionary, token_times, filler_overrides)
    fillers = filler_detector.summarize_fillers(matches, total)
    repetition = repetition_analyser.analyse_repetition(
        tokens, token_times, settings.phrase_min_len, settings.phrase_max_len,
        settings.min_phrase_repeats, settings.overuse_min_count, settings.overuse_min_pct,
    )
    fluency = fluency_analyser.analyse_fluency(
        total, duration, segments, count_words, intervals, silences,
        fillers["per_100_words"], repetition["events"], settings,
    )
    warnings += fluency["warnings"]

    return {
        "tokens": tokens,
        "spans": spans,
        "duration": duration,
        "language": result.language if result else None,
        "timestamp_precision": precision,
        "edited": edited,
        "stats": word_counter.text_stats(text, tokens, len(segments)),
        "fillers": fillers,
        "filler_matches": matches,
        "vocabulary": vocabulary_analyser.analyse_vocabulary(tokens),
        "repetition": repetition,
        "fluency": fluency,
        "warnings": warnings,
    }
