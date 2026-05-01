"""
nb-whisper-large transcriber (Anton fork).

Replaces the upstream OpenAI/local-Whisper paths with a direct HTTP call to
the nb-whisper-large dual-host whisper.cpp service running on Anton:
  - http://192.168.10.11:2025/v1/audio/transcriptions  (Dinesh)
  - http://192.168.10.10:2025/v1/audio/transcriptions  (Gilfoyle, fallback)

The original Captacity caller expects:
  [{"start": float, "end": float, "words": [{"word": str, "start": float, "end": float}, ...]}, ...]

whisper.cpp's OpenAI-compatible response (`verbose_json` + word timestamps) maps
to this shape with minor key normalization.
"""

import os
import requests

# Endpoints can be overridden via env so the same script works on Dinesh-only or Gilfoyle-only hosts.
NB_ENDPOINTS = [
    os.environ.get("NB_WHISPER_PRIMARY", "http://192.168.10.11:2025/v1/audio/transcriptions"),
    os.environ.get("NB_WHISPER_FALLBACK", "http://192.168.10.10:2025/v1/audio/transcriptions"),
]

NB_LANGUAGE = os.environ.get("NB_WHISPER_LANGUAGE", "no")
NB_MODEL = os.environ.get("NB_WHISPER_MODEL", "nb-whisper-large")
NB_TIMEOUT_SEC = int(os.environ.get("NB_WHISPER_TIMEOUT", "600"))


def _normalize_segments(payload):
    segments = payload.get("segments")
    if not segments:
        # Some whisper.cpp builds return only "text" + "words" at the top level.
        words = payload.get("words", [])
        if not words:
            return []
        return [{
            "start": words[0].get("start", 0.0),
            "end": words[-1].get("end", words[0].get("end", 0.0)),
            "words": [_normalize_word(w) for w in words],
        }]
    out = []
    for seg in segments:
        words = seg.get("words", [])
        out.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "words": [_normalize_word(w) for w in words],
        })
    return out


def _normalize_word(w):
    word = w.get("word", w.get("text", ""))
    if word and not word.startswith(" "):
        word = " " + word
    return {
        "word": word,
        "start": float(w.get("start", 0.0)),
        "end": float(w.get("end", 0.0)),
    }


def transcribe_with_api(audio_file, prompt=None):
    """Transcribe audio via nb-whisper-large (anton fork)."""
    last_error = None
    for endpoint in NB_ENDPOINTS:
        try:
            with open(audio_file, "rb") as fh:
                files = {"file": (os.path.basename(audio_file), fh, "audio/wav")}
                data = {
                    "model": NB_MODEL,
                    "language": NB_LANGUAGE,
                    "response_format": "verbose_json",
                    "timestamp_granularities": "word",
                }
                if prompt:
                    data["prompt"] = prompt
                resp = requests.post(endpoint, files=files, data=data, timeout=NB_TIMEOUT_SEC)
            resp.raise_for_status()
            return _normalize_segments(resp.json())
        except (requests.RequestException, ValueError) as e:
            last_error = e
            continue
    raise RuntimeError(f"All nb-whisper endpoints failed: {last_error}")


def transcribe_locally(audio_file, prompt=None):
    """Backward-compat alias — same impl since we always go through nb-whisper now."""
    return transcribe_with_api(audio_file, prompt=prompt)
