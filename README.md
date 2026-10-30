# Speech Analyser

Local Streamlit app: upload a recording, get a transcript, word count, WPM, fillers,
vocabulary, repetition and pause/fluency analysis, and export TXT / JSON / CSV.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional: better content-word detection
streamlit run app.py
```
Audio is decoded with PyAV (installed with `faster-whisper`), so system ffmpeg is not required.
The first run downloads the chosen Whisper model (tiny ≈ 75 MB … medium ≈ 1.5 GB); afterwards
everything runs offline and nothing leaves your computer.

## Docker (opens in the browser, no Python setup)
```bash
docker compose up -d --build     # or double-click start.command on macOS
```
Then open http://localhost:8600. The image includes the `base` Whisper model and spaCy's
English model, so it works offline after the build. Choose another model with
`WHISPER_MODEL` in `docker-compose.yml`. Stop with `docker compose down`.

## Layout
`app.py` is UI only. Each analysis lives in `services/` (word_counter, filler_detector,
vocabulary_analyser, repetition_analyser, fluency_analyser) and is combined by `services/pipeline.py`.
Tokenisation rules are shared via `utils/text_utils.py`. Run tests with `python -m pytest`.

## Notes
- Filler counts and the fluency score are estimates, not validated assessments.
- Pauses come from audio silence detection (thresholds in the sidebar); leading/trailing silence is ignored by default.
- Editing the transcript recalculates text metrics; word timestamps are then unavailable.
