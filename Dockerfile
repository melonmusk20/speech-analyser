FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8600

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Bake a Whisper model into the image so the app works offline right after install.
ARG WHISPER_MODEL=base
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

COPY . .

EXPOSE 8600
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8600/_stcore/health')"

CMD ["streamlit", "run", "app.py"]
