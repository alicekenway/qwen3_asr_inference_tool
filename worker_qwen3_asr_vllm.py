#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Independent Qwen3-ASR + vLLM worker.

This program intentionally knows nothing about Slurm. It reads a candidate
manifest JSONL and appends ASR labels to an output JSONL file, so the same
worker can later be launched by Slurm, a local process pool, Ray, Kubernetes,
or any other scheduler.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from common import append_jsonl_record, extract_asr_result, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-ASR vLLM manifest worker")
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest for this worker shard.")
    parser.add_argument("--output", required=True, help="Output JSONL labels for this worker shard.")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--language", default=None)
    parser.add_argument("--context", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--vllm-attention-backend",
        default="FLASH_ATTN",
        help="Set VLLM_ATTENTION_BACKEND before importing vLLM. Use 'auto' to leave it unset.",
    )
    parser.add_argument(
        "--flashinfer-disable-version-check",
        choices=("0", "1"),
        default="1",
        help=(
            "Set FLASHINFER_DISABLE_VERSION_CHECK before importing vLLM. "
            "Default 1 avoids vLLM startup failures caused by mismatched flashinfer/flashinfer-cubin packages."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing successful records in --output.")
    return parser.parse_args()


def successful_keys(output_path: Path) -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    if not output_path.exists():
        return done
    for item in load_jsonl(output_path):
        if item.get("error") is None and item.get("id") is not None and item.get("candidate_index") is not None:
            done.add((str(item["id"]), int(item["candidate_index"])))
    return done


def batched(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def _model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    return kwargs


def build_model(args: argparse.Namespace) -> Any:
    if args.backend == "vllm" and args.vllm_attention_backend.lower() != "auto":
        os.environ["VLLM_ATTENTION_BACKEND"] = args.vllm_attention_backend
    if args.backend == "vllm":
        os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = args.flashinfer_disable_version_check

    from qwen_asr import Qwen3ASRModel

    if args.backend == "vllm":
        if not hasattr(Qwen3ASRModel, "LLM"):
            raise TypeError("Installed qwen_asr.Qwen3ASRModel does not expose the vLLM LLM() factory")
        return Qwen3ASRModel.LLM(
            model=args.model,
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            **_model_kwargs(args),
        )

    if not hasattr(Qwen3ASRModel, "from_pretrained"):
        raise TypeError("Installed qwen_asr.Qwen3ASRModel does not expose from_pretrained()")
    return Qwen3ASRModel.from_pretrained(
        args.model,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


def _call_method(method: Any, audios: List[str], args: argparse.Namespace) -> Any:
    kwargs: Dict[str, Any] = {}
    if args.language:
        kwargs["language"] = args.language
    if args.context:
        kwargs["context"] = args.context

    attempts = [
        (audios, kwargs),
        (audios, {}),
    ]
    last_error: Optional[TypeError] = None
    for audio_arg, call_kwargs in attempts:
        try:
            return method(audio_arg, **call_kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("ASR method call failed without an exception")


def transcribe_batch(model: Any, audios: List[str], args: argparse.Namespace) -> List[Any]:
    for method_name in ("transcribe", "generate", "__call__"):
        if not hasattr(model, method_name):
            continue
        method = getattr(model, method_name)
        try:
            results = _call_method(method, audios, args)
            if isinstance(results, list):
                return results
            if isinstance(results, tuple):
                return list(results)
            return [results]
        except TypeError:
            pass

    # Conservative fallback for wrappers that only support one audio at a time.
    results = []
    method = getattr(model, "transcribe", None) or getattr(model, "generate", None) or model
    for audio in audios:
        results.append(_call_method(method, [audio], args))
    return results


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(load_jsonl(manifest_path))
    done = set() if args.overwrite else successful_keys(output_path)
    todo = [
        item
        for item in records
        if (str(item.get("id")), int(item.get("candidate_index"))) not in done
    ]

    if not todo:
        print(f"[worker] nothing to do: {manifest_path}")
        return 0

    print(
        f"[worker] pid={os.getpid()} manifest={manifest_path} total={len(records)} "
        f"done={len(done)} todo={len(todo)} output={output_path}"
    )
    model = build_model(args)

    mode = "w" if args.overwrite else "a"
    with output_path.open(mode, encoding="utf-8") as fout:
        for batch in tqdm(
            list(batched(todo, args.batch_size)),
            desc=f"label {manifest_path.name}",
            file=sys.stdout,
            dynamic_ncols=True,
        ):
            audios = [str(item["audio"]) for item in batch]
            try:
                results = transcribe_batch(model, audios, args)
                if len(results) != len(batch):
                    if len(batch) == 1:
                        results = [results[0] if results else ""]
                    else:
                        raise RuntimeError(f"ASR returned {len(results)} results for batch size {len(batch)}")
            except Exception as exc:
                for item in batch:
                    append_jsonl_record(
                        fout,
                        {
                            **item,
                            "asr_text": "",
                            "language": args.language,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                continue

            for item, result in zip(batch, results):
                text, language, extra = extract_asr_result(result)
                record = {
                    **item,
                    "asr_text": text,
                    "language": language if language is not None else args.language,
                    "error": None,
                }
                if extra:
                    record["asr_extra"] = extra
                append_jsonl_record(fout, record)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
