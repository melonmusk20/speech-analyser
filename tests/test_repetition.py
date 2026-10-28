from services.repetition_analyser import analyse_repetition, find_consecutive_repeats
from utils.text_utils import tokenize


def test_consecutive_word():
    r = find_consecutive_repeats(tokenize("I I agree"))
    assert len(r) == 1 and r[0]["phrase"] == "i" and r[0]["repeats"] == 2


def test_consecutive_phrase_ignores_punctuation():
    r = find_consecutive_repeats(tokenize("I was going, I was going to leave"))
    assert r[0]["phrase"] == "i was going" and r[0]["type"].startswith("repeated phrase")


def test_non_consecutive_phrases_and_subsumption():
    toks = tokenize("machine learning models are fun. I study machine learning models daily")
    phrases = analyse_repetition(toks)["phrases"]
    assert [p["phrase"] for p in phrases] == ["machine learning models"]


def test_stopword_only_phrases_ignored():
    toks = tokenize("it is the one. it is the other")
    assert analyse_repetition(toks)["phrases"] == []


def test_overused_word():
    toks = tokenize("data " * 6 + "and other words here to pad the transcript a bit more " * 1)
    assert analyse_repetition(toks, overuse_min_pct=5)["overused"][0]["word"] == "data"


def test_empty():
    assert analyse_repetition([])["events"] == 0
