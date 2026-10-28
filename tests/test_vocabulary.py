from services.vocabulary_analyser import analyse_vocabulary, moving_average_ttr
from utils.text_utils import tokenize


def test_ttr_and_length():
    v = analyse_vocabulary(tokenize("one two two three"))
    assert v["total_words"] == 4 and v["unique_words"] == 3
    assert v["ttr"] == 75.0
    assert v["avg_word_length"] == 3.5  # 14 letters / 4 words


def test_empty():
    assert analyse_vocabulary([])["total_words"] == 0


def test_mattr_all_unique_and_all_same():
    assert moving_average_ttr([str(i) for i in range(60)]) == 100.0
    assert moving_average_ttr(["a"] * 60) == 100 / 50
    assert moving_average_ttr(["a"] * 10) is None
