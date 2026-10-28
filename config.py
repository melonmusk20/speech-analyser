"""Central configuration: limits, model options and tunable analysis settings."""
from dataclasses import dataclass

APP_NAME = "Speech Analyser"
SUPPORTED_EXTENSIONS = ("mp3", "wav", "m4a", "aac", "flac", "mp4")
MAX_FILE_MB = 500
MAX_DURATION_SEC = 3 * 60 * 60
MIN_DURATION_SEC = 0.5

# model name -> (approx download, approx RAM, note)
MODEL_OPTIONS = {
    "tiny": "~75 MB download, ~1 GB RAM. Fastest, least accurate.",
    "base": "~145 MB download, ~1 GB RAM. Good for quick checks.",
    "small": "~485 MB download, ~2 GB RAM. Balanced (recommended).",
    "medium": "~1.5 GB download, ~5 GB RAM. More accurate, slower.",
}
DEFAULT_MODEL = "base"

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Hindi": "hi",
    "Japanese": "ja",
    "Chinese": "zh",
}

# Silence detection (energy based): a frame is silent when its RMS is more than
# SILENCE_DB decibels below the 95th-percentile RMS of the recording.
SILENCE_DB = 35.0
FRAME_SEC = 0.02
MIN_SILENCE_SEC = 0.15

# Value at which each fluency-score component reaches its full penalty.
FLUENCY_FULL_PENALTY = {
    "rate_consistency": 0.5,  # coefficient of variation of segment WPM
    "fillers": 10.0,  # fillers per 100 words
    "long_pauses": 4.0,  # long pauses per minute
    "repetition": 8.0,  # repetition events per 100 words
}


@dataclass(frozen=True)
class AnalysisSettings:
    short_pause_min: float = 0.3
    long_pause_min: float = 1.0
    include_edge_silence: bool = False
    phrase_min_len: int = 2
    phrase_max_len: int = 8
    min_phrase_repeats: int = 2
    overuse_min_count: int = 5
    overuse_min_pct: float = 2.0
    slow_wpm: float = 110.0
    fast_wpm: float = 160.0
    # (component, weight) pairs for the experimental fluency score
    weights: tuple = (
        ("rate_consistency", 20),
        ("fillers", 30),
        ("long_pauses", 30),
        ("repetition", 20),
    )
