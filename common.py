#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for large-scale Qwen3 ASR candidate labeling."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: Any, max_prefix: int = 120) -> str:
    raw = str(value)
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    if not prefix:
        prefix = "record"
    prefix = prefix[:max_prefix]
    digest = sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def json_dumps(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def iter_json_or_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield records from a JSON array, a single JSON object, or JSONL.

    For very large JSON arrays, installing ``ijson`` enables streaming. Without
    it, standard-library ``json.load`` is used for array inputs.
    """
    with path.open("r", encoding="utf-8") as f:
        head = f.read(4096)

    stripped = head.lstrip()
    if not stripped:
        return

    if stripped.startswith("["):
        try:
            import ijson  # type: ignore
        except Exception:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} starts with '[' but did not parse as a JSON array")
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(f"{path} contains a non-object array item: {type(item).__name__}")
                yield item
            return

        with path.open("rb") as f:
            for item in ijson.items(f, "item"):
                if not isinstance(item, dict):
                    raise ValueError(f"{path} contains a non-object array item: {type(item).__name__}")
                yield item
        return

    # Prefer JSONL for object-prefixed files. If that fails because the file is
    # pretty-printed single-object JSON, fall back to json.load.
    try:
        with path.open("r", encoding="utf-8") as f:
            yielded = False
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                yielded = True
                yield item
            if yielded:
                return
    except json.JSONDecodeError:
        pass

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"{path} contains a non-object array item: {type(item).__name__}")
            yield item
    else:
        raise ValueError(f"{path} must contain a JSON object, JSON array, or JSONL objects")


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: line is not an object")
            yield item


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json_dumps(record) + "\n")
            count += 1
    return count


def append_jsonl_record(fout: Any, record: Dict[str, Any]) -> None:
    fout.write(json_dumps(record) + "\n")
    fout.flush()


def resolve_audio_path(raw_path: str, source_json: Path, audio_root: Optional[Path]) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if audio_root is not None:
        return audio_root / path
    return source_json.parent / path


def _resample_1d(audio: "Any", source_sr: int, target_sr: int) -> "Any":
    if source_sr == target_sr:
        return audio

    try:
        from scipy.signal import resample_poly  # type: ignore

        gcd = math.gcd(source_sr, target_sr)
        return resample_poly(audio, target_sr // gcd, source_sr // gcd).astype("float32")
    except Exception:
        import numpy as np

        if len(audio) == 0:
            return audio.astype("float32")
        target_len = max(1, int(round(len(audio) * float(target_sr) / float(source_sr))))
        old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        return np.interp(new_x, old_x, audio).astype("float32")


def read_audio_mono_16k(path: Path, target_sr: int = 16000) -> "Any":
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=True)
    if audio.size == 0:
        return np.zeros((0,), dtype="float32")
    audio = audio.astype("float32", copy=False)
    mono = audio.mean(axis=1)
    return _resample_1d(mono, int(sr), int(target_sr))


def concat_audio_files(
    input_paths: Sequence[Path],
    output_path: Path,
    target_sr: int = 16000,
    gap_ms: float = 0.0,
) -> None:
    import numpy as np
    import soundfile as sf

    if not input_paths:
        raise ValueError("Cannot concatenate an empty audio path list")

    chunks = []
    gap = np.zeros((max(0, int(round(target_sr * gap_ms / 1000.0))),), dtype="float32")
    for idx, path in enumerate(input_paths):
        if not path.exists():
            raise FileNotFoundError(str(path))
        if idx and len(gap):
            chunks.append(gap)
        chunks.append(read_audio_mono_16k(path, target_sr=target_sr))

    audio = np.concatenate(chunks).astype("float32", copy=False) if chunks else np.zeros((0,), dtype="float32")
    ensure_dir(output_path.parent)
    sf.write(str(output_path), audio, int(target_sr), subtype="PCM_16")


def normalize_text_basic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    chars = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def tokenize_for_wer(text: str, normalize: str = "basic", tokenizer: str = "word") -> List[str]:
    if normalize == "basic":
        text = normalize_text_basic(text)
    elif normalize == "none":
        text = " ".join(text.split())
    else:
        raise ValueError(f"Unknown WER normalization: {normalize}")

    if tokenizer == "word":
        tokens = text.split()
        if tokens:
            return tokens
        return list(text)
    if tokenizer == "char":
        return [ch for ch in text if not ch.isspace()]
    raise ValueError(f"Unknown WER tokenizer: {tokenizer}")


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hyp_token in enumerate(hyp, start=1):
            cost = 0 if ref_token == hyp_token else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str, normalize: str = "basic", tokenizer: str = "word") -> float:
    ref_tokens = tokenize_for_wer(reference, normalize=normalize, tokenizer=tokenizer)
    hyp_tokens = tokenize_for_wer(hypothesis, normalize=normalize, tokenizer=tokenizer)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return float(edit_distance(ref_tokens, hyp_tokens)) / float(len(ref_tokens))


def extract_asr_result(result: Any) -> Tuple[str, Optional[str], Dict[str, Any]]:
    if isinstance(result, dict):
        text = str(result.get("text", "") or "")
        language = result.get("language")
        extra = {k: v for k, v in result.items() if k not in {"text", "language"}}
        return text, language, extra

    text = str(getattr(result, "text", "") or "")
    language = getattr(result, "language", None)
    extra: Dict[str, Any] = {}
    if hasattr(result, "__dict__"):
        extra = {k: v for k, v in vars(result).items() if k not in {"text", "language"}}
    return text, language, extra

