"""Audio validation, decoding and silence detection (FR-01, FR-08)."""
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

SAMPLE_RATE = 16000
TEMP_DIR = Path(tempfile.gettempdir()) / "speech_analyser"


class AudioError(Exception):
    """A user-presentable problem with the audio file."""


@dataclass
class AudioInfo:
    duration: float
    silences: list[tuple[float, float]] = field(default_factory=list)
    silent_fraction: float = 0.0


def validate_upload(name: str, size_bytes: int) -> None:
    ext = Path(name).suffix.lstrip(".").lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise AudioError(
            f"'.{ext or '?'}' files are not supported. Use one of: "
            + ", ".join(e.upper() for e in config.SUPPORTED_EXTENSIONS) + "."
        )
    if size_bytes <= 0:
        raise AudioError("The file is empty.")
    if size_bytes > config.MAX_FILE_MB * 1024 * 1024:
        raise AudioError(f"The file is larger than the {config.MAX_FILE_MB} MB limit.")


def save_upload(data: bytes, name: str) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-]", "_", Path(name).name) or "recording"
    path = TEMP_DIR / safe
    path.write_bytes(data)
    return path


def clear_temp_files() -> None:
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode to mono float32. Uses PyAV (bundled with faster-whisper), then librosa."""
    errors = []
    try:
        from faster_whisper.audio import decode_audio

        return np.asarray(decode_audio(str(path), sampling_rate=sr), dtype=np.float32)
    except ImportError:
        errors.append("faster-whisper is not installed")
    except Exception as exc:
        errors.append(str(exc))
    try:
        import librosa

        samples, _ = librosa.load(str(path), sr=sr, mono=True)
        return samples.astype(np.float32)
    except ImportError:
        errors.append("librosa is not installed")
    except Exception as exc:
        errors.append(str(exc))
    raise AudioError(
        "This file could not be decoded; it may be corrupted or in an unsupported format. "
        f"Details: {'; '.join(errors)}"
    )


def find_silences(samples: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[list[tuple[float, float]], float]:
    """Return (silent intervals in seconds, peak-reference RMS)."""
    hop = int(sr * config.FRAME_SEC)
    n = len(samples) // hop
    if n == 0:
        return [], 0.0
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    ref = float(np.percentile(rms, 95))
    if ref < 1e-4:
        return [(0.0, len(samples) / sr)], ref
    silent = rms < ref * 10 ** (-config.SILENCE_DB / 20)
    padded = np.concatenate(([False], silent, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    min_frames = config.MIN_SILENCE_SEC / config.FRAME_SEC
    return [
        (float(s * config.FRAME_SEC), float(e * config.FRAME_SEC))
        for s, e in zip(edges[::2], edges[1::2])
        if e - s >= min_frames
    ], ref


def analyse_audio(path: str | Path) -> AudioInfo:
    samples = load_audio(path)
    duration = len(samples) / SAMPLE_RATE
    if duration < config.MIN_DURATION_SEC:
        raise AudioError("The recording is too short to analyse.")
    if duration > config.MAX_DURATION_SEC:
        raise AudioError(f"The recording is longer than {config.MAX_DURATION_SEC // 3600} hours.")
    silences, ref = find_silences(samples)
    if ref < 1e-4:
        raise AudioError("The recording appears to be silent: no readable audio was found.")
    silent_total = sum(e - s for s, e in silences)
    return AudioInfo(duration=duration, silences=silences, silent_fraction=silent_total / duration)
