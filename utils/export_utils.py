"""TXT / JSON / CSV exports."""
import csv
import io
import json
from datetime import datetime

from services.word_counter import word_frequencies

_DROP = {"tokens", "spans", "filler_matches"}


def transcript_txt(text: str) -> str:
    return text


def build_report(analysis: dict, text: str, source_name: str = "") -> dict:
    """JSON-safe report: analysis minus bulky internals, plus the transcript."""
    rep = {k: v for k, v in analysis.items() if k not in _DROP}
    rep["repetition"] = {k: v for k, v in analysis["repetition"].items() if k != "highlight_ranges"}
    rep["fillers"] = analysis["fillers"] | {"matches": analysis["filler_matches"]}
    rep["top_words"] = word_frequencies(analysis["tokens"], 50, exclude_stop_words=True)
    rep["transcript"] = text
    rep["meta"] = {
        "source_file": source_name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "transcript_edited": analysis["edited"],
        "note": "Speech-recognition output and filler/fluency figures are estimates.",
    }
    return rep


def report_json(analysis: dict, text: str, source_name: str = "") -> str:
    return json.dumps(build_report(analysis, text, source_name), indent=2, ensure_ascii=False, default=str)


def frequencies_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["word", "count", "percentage"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()
