"""Speech Analyser: Streamlit UI. All analysis logic lives in services/."""
import sys
from pathlib import Path

from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx(suppress_warning=True) is None:  # started with `python3 app.py`
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--server.port", "8600"]
    sys.exit(cli.main())

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from services import audio_processor as ap
from services import pipeline, transcription as tr
from services.filler_detector import DEFAULT_DICTIONARY
from services.word_counter import search_word, word_frequencies
from utils import export_utils
from utils.text_utils import format_time, highlight_html

st.set_page_config(page_title=config.APP_NAME, page_icon="🎙️", layout="wide")
_css = Path(__file__).parent / "assets" / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

ss = st.session_state
for key, default in {"result": None, "audio": None, "file_key": None, "edited_text": None,
                     "fill_overrides": {}, "error": None, "source_name": ""}.items():
    ss.setdefault(key, default)


def fmt(v, digits=0, unit=""):
    return "–" if v is None else f"{v:,.{digits}f}{unit}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Transcription model", list(config.MODEL_OPTIONS), index=list(config.MODEL_OPTIONS).index(config.DEFAULT_MODEL))
    st.caption(config.MODEL_OPTIONS[model])
    language_label = st.selectbox("Language", list(config.LANGUAGES))

    with st.expander("Pause & fluency settings"):
        short_p = st.slider("Short pause from (s)", 0.15, 1.0, 0.3, 0.05)
        long_p = st.slider("Long pause from (s)", 0.5, 3.0, 1.0, 0.1)
        edges = st.checkbox("Include leading/trailing silence", False)
        phrase_range = st.slider("Repeated phrase length (words)", 2, 8, (2, 8))
        min_rep = st.number_input("Min. phrase repetitions", 2, 10, 2)
    with st.expander("Filler dictionary"):
        definite = st.text_area("Always fillers (one per line)", "\n".join(DEFAULT_DICTIONARY["definite"]))
        context = st.text_area("Context-dependent (one per line)", "\n".join(DEFAULT_DICTIONARY["context"]))

    st.divider()
    st.caption("🔒 Audio and transcripts are processed on this computer. Internet is only needed "
               "once to download the transcription model.")
    if st.button("Clear temporary files"):
        ap.clear_temp_files()
        st.toast("Temporary files cleared.")

settings = config.AnalysisSettings(
    short_pause_min=short_p, long_pause_min=max(long_p, short_p), include_edge_silence=edges,
    phrase_min_len=phrase_range[0], phrase_max_len=phrase_range[1], min_phrase_repeats=int(min_rep),
)
filler_dict = {"definite": [l.strip() for l in definite.splitlines() if l.strip()],
               "context": [l.strip() for l in context.splitlines() if l.strip()]}

# ---------------------------------------------------------------- header / upload
st.title("🎙️ Speech Analyser")
st.caption("Your speech, measured and understood.")

upload = st.file_uploader("Upload your recording", type=list(config.SUPPORTED_EXTENSIONS),
                          help="MP3 · WAV · M4A · AAC · FLAC · MP4")


def run_analysis(file):
    ss.error = None
    bar = st.progress(0.0, "Checking file…")
    try:
        ap.validate_upload(file.name, file.size)
        path = ap.save_upload(file.getvalue(), file.name)
        bar.progress(0.05, "Reading audio and detecting silences…")
        audio = ap.analyse_audio(path)
        bar.progress(0.1, "Loading model (first run downloads it)…")
        tr.load_model(model)
        result = tr.transcribe(
            path, model, config.LANGUAGES[language_label], audio.duration,
            lambda f: bar.progress(0.1 + 0.88 * f, f"Transcribing… {f:.0%}"),
        )
        ss.update(result=result, audio=audio, edited_text=None, fill_overrides={},
                  file_key=(file.name, file.size, model, language_label), source_name=file.name)
        bar.progress(1.0, "Done")
    except (ap.AudioError, tr.TranscriptionError) as exc:
        ss.error = str(exc)
        bar.empty()
    except Exception as exc:  # never crash the UI
        ss.error = f"Unexpected error: {exc}"
        bar.empty()


if upload is not None:
    key = (upload.name, upload.size, model, language_label)
    c1, c2 = st.columns([3, 1])
    c1.write(f"**{upload.name}** · {upload.size / 1e6:.1f} MB")
    label = "Retry analysis" if ss.error else ("Re-run analysis" if ss.file_key == key else "Transcribe & analyse")
    if c2.button(label, type="primary", use_container_width=True):
        run_analysis(upload)
        st.rerun() if not ss.error else None
    if ss.file_key not in (None, key) and ss.error is None:
        st.info("Settings or file changed since the last analysis. Press the button to analyse again.")

if ss.error:
    st.error(ss.error)

if ss.result is None:
    if not ss.error:
        st.info("Upload a recording and press **Transcribe & analyse** to begin.")
    st.stop()

# ---------------------------------------------------------------- analysis (cached per input state)
result: tr.TranscriptionResult = ss.result
original_text = result.text
text = ss.edited_text if ss.edited_text is not None else original_text
edited = text != original_text
cache_key = (text, edited, settings, str(filler_dict), tuple(sorted(ss.fill_overrides.items())), ss.file_key)
if ss.get("_analysis_key") != cache_key:
    ss._analysis = pipeline.analyse(text, result, ss.audio.silences, edited, settings, filler_dict, ss.fill_overrides)
    ss._analysis_key = cache_key
A = ss._analysis
F, V, R, S = A["fluency"], A["vocabulary"], A["repetition"], A["stats"]

for w in A["warnings"]:
    st.warning(w)

cards = st.columns(4)
cards[0].metric("Total words", f"{S['words']:,}")
cards[1].metric("Speaking rate", fmt(F["overall_wpm"], 0, " WPM"))
cards[2].metric("Unique words", f"{S['unique_words']:,}")
cards[3].metric("Fillers (estimate)", f"{A['fillers']['total']:,}")
st.caption(f"Duration {format_time(A['duration'])} · language: {A['language'] or 'unknown'}"
           + (" · transcript edited" if edited else ""))

tab_t, tab_f, tab_fill, tab_v, tab_r = st.tabs(["Transcript", "Fluency", "Fillers", "Vocabulary", "Repetition"])

# ---------------------------------------------------------------- transcript
with tab_t:
    st.markdown(f"<span class='badge'>{'Edited transcript' if edited else 'Original transcription'}</span>",
                unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Words", S["words"])
    m[1].metric("Sentences", S["sentences"])
    m[2].metric("Paragraphs", S["paragraphs"])
    m[3].metric("Segments", S["segments"])
    st.caption("Speech recognition is not perfectly accurate; review the text and correct it below if needed.")

    show_hl = st.checkbox("Highlight repetitions")
    if show_hl:
        st.markdown(f"<div class='transcript-box'>{highlight_html(text, A['spans'], R['highlight_ranges'])}</div>",
                    unsafe_allow_html=True)
    with st.form("edit"):
        new_text = st.text_area("Transcript (editable)", text, height=300)
        b1, b2 = st.columns(2)
        apply = b1.form_submit_button("Apply edits & recalculate", type="primary")
        revert = b2.form_submit_button("Revert to original")
    if apply:
        ss.edited_text = new_text
        ss.fill_overrides = {}
        st.rerun()
    if revert:
        ss.edited_text, ss.fill_overrides = None, {}
        st.rerun()

    with st.expander("Copy transcript"):
        st.code(text, language=None, wrap_lines=True)
    d = st.columns(3)
    stem = Path(ss.source_name).stem or "recording"
    d[0].download_button("⬇ Transcript (.txt)", text, f"{stem}_transcript.txt", use_container_width=True)
    d[1].download_button("⬇ Report (.json)", export_utils.report_json(A, text, ss.source_name),
                         f"{stem}_report.json", "application/json", use_container_width=True)
    d[2].download_button("⬇ Word frequencies (.csv)",
                         export_utils.frequencies_csv(word_frequencies(A["tokens"], 1000)),
                         f"{stem}_frequencies.csv", "text/csv", use_container_width=True)

# ---------------------------------------------------------------- fluency
with tab_f:
    m = st.columns(4)
    m[0].metric("Overall WPM", fmt(F["overall_wpm"], 0), help="Total words ÷ full recording duration × 60")
    m[1].metric("Active-speech WPM", fmt(F["active_wpm"], 0), help="Words ÷ speaking time, excluding detected pauses")
    m[2].metric("Pauses", F["pause_count"], help=f"≥ {settings.short_pause_min}s · source: {F['pause_source']}")
    m[3].metric("Long pauses", F["long_pause_count"], help=f"≥ {settings.long_pause_min}s")
    st.caption(F["interpretation"])
    m = st.columns(3)
    m[0].metric("Longest pause", fmt(F["longest_pause"], 1, " s"))
    m[1].metric("Words between pauses (avg)", fmt(F["avg_words_between_pauses"], 1))
    m[2].metric("Filler rate", f"{A['fillers']['per_100_words']:.1f} / 100 words")

    seg = pd.DataFrame([s for s in F["segment_wpm"] if s["wpm"]])
    if not seg.empty:
        seg["mid"] = (seg["start"] + seg["end"]) / 2
        fig = px.line(seg, x="mid", y="wpm", markers=True, labels={"mid": "Time (s)", "wpm": "Words per minute"},
                      title="Speaking speed over time (per segment)")
        if F["overall_wpm"]:
            fig.add_hline(y=F["overall_wpm"], line_dash="dot", annotation_text="overall")
        st.plotly_chart(fig, use_container_width=True)
    if F["pauses"]:
        pf = pd.DataFrame(F["pauses"])
        fig = go.Figure(go.Bar(x=pf["start"] + pf["duration"] / 2, y=pf["duration"], width=pf["duration"].clip(lower=0.3),
                               marker_color=["#d9534f" if k == "long" else "#7a9cc6" for k in pf["kind"]]))
        fig.update_layout(title="Pauses (red = long)", xaxis_title="Time (s)", yaxis_title="Duration (s)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({"Start": pf["start"].map(format_time), "Duration (s)": pf["duration"].round(2),
                                   "Type": pf["kind"]}), hide_index=True, use_container_width=True)
    else:
        st.info("No pauses above the short-pause threshold were found.")

    sc = F["score"]
    st.subheader("Experimental fluency score")
    if sc["score"] is None:
        st.info("Not enough data to compute a score.")
    else:
        st.metric("Score (0–100)", sc["score"])
        st.dataframe(pd.DataFrame(sc["components"]), hide_index=True, use_container_width=True)
    st.caption(sc["disclaimer"])

# ---------------------------------------------------------------- fillers
with tab_fill:
    FI = A["fillers"]
    m = st.columns(3)
    m[0].metric("Estimated fillers", FI["total"])
    m[1].metric("Per 100 words", f"{FI['per_100_words']:.1f}")
    m[2].metric("Context words excluded", FI["context_excluded"])
    st.caption(FI["limitations"])
    if FI["by_phrase"]:
        st.plotly_chart(px.bar(x=list(FI["by_phrase"]), y=list(FI["by_phrase"].values()),
                               labels={"x": "Filler", "y": "Count"}), use_container_width=True)
    matches = A["filler_matches"]
    if matches:
        st.subheader("Review")
        st.caption("Untick a row to exclude it, or tick a context-dependent word to count it.")
        df = pd.DataFrame({
            "Include": [m["included"] for m in matches], "Filler": [m["phrase"] for m in matches],
            "Type": ["always" if m["type"] == "definite" else "context-dependent" for m in matches],
            "Time": [format_time(m["time"]) for m in matches], "Context": [m["context"] for m in matches],
        })
        edited_df = st.data_editor(df, hide_index=True, use_container_width=True,
                                   disabled=["Filler", "Type", "Time", "Context"], key=f"fill_{hash(cache_key[:2])}")
        new_over = {m["index"]: bool(inc) for m, inc in zip(matches, edited_df["Include"]) if bool(inc) != m["included"]}
        if new_over:
            ss.fill_overrides = {**ss.fill_overrides, **new_over}
            st.rerun()
    else:
        st.info("No filler words detected. Remember that recognition models may omit hesitation sounds.")

# ---------------------------------------------------------------- vocabulary (+ word frequency)
with tab_v:
    m = st.columns(4)
    m[0].metric("Total / unique words", f"{V['total_words']:,} / {V['unique_words']:,}")
    m[1].metric("Type-token ratio", f"{V['ttr']:.1f}%")
    m[2].metric("Avg. word length", f"{V['avg_word_length']:.2f}")
    m[3].metric("Content words", f"{V['content_word_pct']:.1f}%")
    st.caption(V["note"] + (f" Length-adjusted (moving-average) TTR: {V['mattr']:.1f}%." if V["mattr"] else ""))
    st.caption(f"Content words determined by: {V['method']}.")
    if V["top_content_words"]:
        cw = pd.DataFrame(V["top_content_words"])
        st.plotly_chart(px.bar(cw, x="word", y="count", title="Most frequent content words"), use_container_width=True)
    if V["uncommon_words"]:
        st.write("**Longer, less-repeated words** (simple heuristic):", ", ".join(V["uncommon_words"]))

    st.subheader("Most frequent words")
    c = st.columns([1, 1, 2])
    top_n = c[0].radio("Show", [10, 25, 50], horizontal=True)
    no_stop = c[1].toggle("Exclude stop words", False)
    q = c[2].text_input("Search a word")
    if q:
        r = search_word(A["tokens"], q)
        st.info(f"“{r['word']}” appears **{r['count']}** time(s) ({r['percentage']}% of words).")
    freq = word_frequencies(A["tokens"], top_n, no_stop)
    if freq:
        fdf = pd.DataFrame(freq)
        st.plotly_chart(px.bar(fdf, x="word", y="count"), use_container_width=True)
        st.dataframe(fdf.rename(columns=str.title).assign(Percentage=fdf["percentage"].map("{:.1f}%".format)),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- repetition
with tab_r:
    st.caption(R.get("note", ""))
    st.metric("Repetition events", R["events"])
    if R["consecutive"]:
        st.subheader("Immediate repeats")
        st.dataframe(pd.DataFrame([{
            "Repeated": c["phrase"], "Times": c["repeats"], "Type": c["type"], "Time": format_time(c["time"]),
            "Note": "possible emphasis" if c["possible_emphasis"] else ""} for c in R["consecutive"]]),
            hide_index=True, use_container_width=True)
    if R["phrases"]:
        st.subheader("Repeated phrases")
        st.dataframe(pd.DataFrame([{
            "Phrase": p["phrase"], "Words": p["words"], "Count": p["count"],
            "Times": ", ".join(format_time(t) for t in p["times"][:6] if t is not None)} for p in R["phrases"]]),
            hide_index=True, use_container_width=True)
    if R["overused"]:
        st.subheader("Unusually frequent words")
        st.dataframe(pd.DataFrame(R["overused"]), hide_index=True, use_container_width=True)
    if not (R["consecutive"] or R["phrases"] or R["overused"]):
        st.success("No notable repetition found with the current settings.")
    else:
        st.subheader("Highlighted transcript")
        st.markdown(f"<div class='transcript-box'>{highlight_html(text, A['spans'], R['highlight_ranges'])}</div>",
                    unsafe_allow_html=True)
