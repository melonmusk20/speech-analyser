from config import AnalysisSettings
from services.fluency_analyser import detect_pauses, fluency_score, overall_wpm, analyse_fluency
from services.transcription import Segment
from utils.text_utils import count_words

S = AnalysisSettings()


def test_wpm_100_words_one_minute():
    assert overall_wpm(100, 60) == 100


def test_wpm_safe_for_zero():
    assert overall_wpm(0, 60) is None and overall_wpm(10, 0) is None


def test_silence_pauses_ignore_edges_by_default():
    sil = [(0.0, 2.0), (5.0, 6.5), (7.0, 7.2), (9.0, 10.0)]
    r = detect_pauses(10.0, None if False else [(2.0, 9.0)], sil, S)
    assert [(p["start"], p["kind"]) for p in r["pauses"]] == [(5.0, "long")]
    assert r["span"] == (2.0, 9.0)
    assert r["speaking_time"] == 7.0 - 1.5


def test_include_edges():
    s = AnalysisSettings(include_edge_silence=True)
    r = detect_pauses(10.0, [(2.0, 9.0)], [(0.0, 2.0), (9.0, 10.0)], s)
    assert len(r["pauses"]) == 2 and r["speaking_time"] == 7.0


def test_gap_fallback_and_warning():
    r = detect_pauses(10.0, [(0, 2), (3.5, 5)], None, S)
    assert r["pauses"][0]["duration"] == 1.5 and r["warnings"]


def test_no_timestamps_warns():
    r = detect_pauses(10.0, [], None, S)
    assert r["source"] == "none" and r["warnings"]


def test_score_skips_missing_and_is_bounded():
    assert fluency_score({}, S)["score"] is None
    perfect = fluency_score({"rate_consistency": 0, "fillers": 0, "long_pauses": 0, "repetition": 0}, S)
    worst = fluency_score({"rate_consistency": 9, "fillers": 99, "long_pauses": 99, "repetition": 99}, S)
    assert perfect["score"] == 100 and worst["score"] == 0


def test_analyse_fluency_end_to_end():
    segs = [Segment(0, 10, "word " * 20), Segment(10, 20, "word " * 20), Segment(20, 30, "word " * 20)]
    f = analyse_fluency(60, 30, segs, count_words, [(0, 30)], [], 0.0, 0, S)
    assert f["overall_wpm"] == 120 and f["rate_variation_cv"] == 0
    assert f["pause_count"] == 0
