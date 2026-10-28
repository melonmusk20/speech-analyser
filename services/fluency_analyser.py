"""Speaking rate, pause detection and the experimental fluency score (FR-03, FR-08)."""
import statistics

from config import FLUENCY_FULL_PENALTY, AnalysisSettings

SCORE_DISCLAIMER = (
    "Experimental heuristic combining speaking-rate consistency, filler rate, long pauses and "
    "repetition. It is an estimate for self-reflection, not a validated linguistic or clinical assessment."
)


def overall_wpm(word_count: int, duration_sec: float) -> float | None:
    if duration_sec <= 0 or word_count <= 0:
        return None
    return word_count / duration_sec * 60


def interpret_wpm(wpm: float | None, settings: AnalysisSettings) -> str:
    if wpm is None:
        return "Not enough data."
    band = "slower" if wpm < settings.slow_wpm else "faster" if wpm > settings.fast_wpm else "moderate"
    return (
        f"{band.capitalize()} pace for conversational speech. This is informational only: the "
        "right pace depends on audience, content and setting."
    )


def segment_wpms(segments, count_words) -> list[dict]:
    out = []
    for seg in segments:
        dur = seg.end - seg.start
        words = count_words(seg.text)
        out.append(
            {
                "start": seg.start,
                "end": seg.end,
                "words": words,
                "wpm": words / dur * 60 if dur > 0 and words else None,
            }
        )
    return out


def _gaps(intervals, duration, settings):
    """Silent gaps between consecutive speech intervals (plus edges if requested)."""
    gaps = []
    intervals = sorted(intervals)
    if settings.include_edge_silence:
        gaps.append((0.0, intervals[0][0]))
    for (_, a_end), (b_start, _) in zip(intervals, intervals[1:]):
        gaps.append((a_end, b_start))
    if settings.include_edge_silence:
        gaps.append((intervals[-1][1], duration))
    span = (0.0, duration) if settings.include_edge_silence else (intervals[0][0], intervals[-1][1])
    return gaps, span


def _silence_gaps(silences, duration, settings):
    edge = 0.05
    inner = [s for s in silences if s[0] > edge and s[1] < duration - edge]
    lead = next((s for s in silences if s[0] <= edge), None)
    trail = next((s for s in silences if s[1] >= duration - edge), None)
    start = 0.0 if settings.include_edge_silence or not lead else lead[1]
    end = duration if settings.include_edge_silence or not trail else trail[0]
    gaps = inner + ([lead] if settings.include_edge_silence and lead else []) + (
        [trail] if settings.include_edge_silence and trail else []
    )
    return gaps, (start, end)


def detect_pauses(duration, speech_intervals, silences, settings: AnalysisSettings) -> dict:
    """Pauses from audio silence detection when available, else timestamp gaps."""
    warnings = []
    if silences is not None:
        gaps, span = _silence_gaps(silences, duration, settings)
        source = "audio silence detection"
    elif speech_intervals:
        gaps, span = _gaps(speech_intervals, duration, settings)
        source = "transcript timestamps"
        warnings.append("Pauses were estimated from transcript timestamps, which can be imprecise.")
    else:
        return {"pauses": [], "source": "none", "span": (0.0, duration), "speaking_time": None,
                "warnings": ["No reliable timestamps: pause statistics are unavailable."]}

    pauses = []
    for s, e in sorted(gaps):
        d = e - s
        if d >= settings.short_pause_min:
            pauses.append({"start": s, "end": e, "duration": d,
                           "kind": "long" if d >= settings.long_pause_min else "short"})
    span_len = max(0.0, span[1] - span[0])
    speaking_time = max(0.0, span_len - sum(p["duration"] for p in pauses if p["start"] >= span[0] - 1e-6 and p["end"] <= span[1] + 1e-6))
    return {"pauses": pauses, "source": source, "span": span, "speaking_time": speaking_time, "warnings": warnings}


def fluency_score(components: dict, settings: AnalysisSettings) -> dict:
    """components: name -> measured value or None. Missing components are skipped."""
    weights = dict(settings.weights)
    used, total_w, penalty = [], 0.0, 0.0
    for name, w in weights.items():
        value = components.get(name)
        if value is None or w <= 0:
            continue
        p = min(1.0, max(0.0, value / FLUENCY_FULL_PENALTY[name]))
        used.append({"component": name, "value": round(value, 3), "penalty": round(p, 3), "weight": w})
        total_w += w
        penalty += w * p
    if not total_w:
        return {"score": None, "components": [], "disclaimer": SCORE_DISCLAIMER}
    return {"score": round(100 * (1 - penalty / total_w)), "components": used, "disclaimer": SCORE_DISCLAIMER}


def analyse_fluency(word_count, duration, segments, count_words, speech_intervals, silences,
                    filler_per_100, repetition_events, settings: AnalysisSettings) -> dict:
    seg_rates = segment_wpms(segments, count_words)
    rates = [s["wpm"] for s in seg_rates if s["wpm"] and s["end"] - s["start"] >= 1 and s["words"] >= 3]
    cv = statistics.pstdev(rates) / statistics.mean(rates) if len(rates) >= 3 else None

    pause_info = detect_pauses(duration, speech_intervals, silences, settings)
    pauses, speaking_time = pause_info["pauses"], pause_info["speaking_time"]
    long_pauses = [p for p in pauses if p["kind"] == "long"]
    span_min = (pause_info["span"][1] - pause_info["span"][0]) / 60
    have_timing = pause_info["source"] != "none"

    overall = overall_wpm(word_count, duration)
    active = overall_wpm(word_count, speaking_time) if speaking_time else None
    rep_per_100 = 100 * repetition_events / word_count if word_count else None
    score = fluency_score(
        {
            "rate_consistency": cv,
            "fillers": filler_per_100 if word_count else None,
            "long_pauses": len(long_pauses) / span_min if have_timing and span_min > 0 else None,
            "repetition": rep_per_100,
        },
        settings,
    )
    return {
        "overall_wpm": overall,
        "active_wpm": active,
        "interpretation": interpret_wpm(overall, settings),
        "segment_wpm": seg_rates,
        "rate_variation_cv": cv,
        "pauses": pauses,
        "pause_count": len(pauses),
        "long_pause_count": len(long_pauses),
        "total_pause_time": sum(p["duration"] for p in pauses),
        "longest_pause": max((p["duration"] for p in pauses), default=0.0),
        "avg_words_between_pauses": word_count / (len(pauses) + 1) if have_timing and word_count else None,
        "speaking_time": speaking_time,
        "pause_source": pause_info["source"],
        "score": score,
        "warnings": pause_info["warnings"],
    }
