from services.filler_detector import detect_fillers, summarize_fillers
from utils.text_utils import tokenize_spans


def run(text, **kw):
    spans = tokenize_spans(text)
    m = detect_fillers(spans, text, **kw)
    return m, summarize_fillers(m, len(spans))


def test_counts_um_twice():
    _, s = run("Um I think um we should go")
    assert s["by_phrase"] == {"um": 2}


def test_multiword_not_double_counted():
    m, s = run("It was kind of, you know, sort of fine")
    assert [x["phrase"] for x in m] == ["kind of", "you know", "sort of"]
    assert s["total"] == 3


def test_context_words_need_context():
    _, s = run("I like pizza. So, I went home")
    assert s["by_phrase"] == {"so": 1}
    assert s["context_excluded"] == 1


def test_manual_override():
    m, _ = run("I like pizza")
    _, s = run("I like pizza", overrides={m[0]["index"]: True})
    assert s["total"] == 1


def test_rate_per_100_and_empty():
    _, s = run("")
    assert s["per_100_words"] == 0.0 and s["total"] == 0
