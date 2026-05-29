#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main program for large-scale Qwen3 candidate labeling with Slurm.

Responsibilities:
  1. Read input JSON/JSONL records.
  2. For each record, concatenate reference text and concatenate candidate
     audio by candidate index, resampled to 16 kHz mono WAV.
  3. Split the candidate-audio manifest across N workers.
  4. Launch independent workers with ``sbatch``.
  5. Collect worker transcripts and compute WER / best candidate.
"""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from common import (
    append_jsonl_record,
    concat_audio_files,
    ensure_dir,
    iter_json_or_jsonl,
    json_dumps,
    load_jsonl,
    resolve_audio_path,
    safe_name,
    wer,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, submit, and collect Qwen3-ASR candidate labels.")
    parser.add_argument("--input-json", "--input_json", nargs="+", required=True, help="Input JSON/JSONL files, dirs, or globs.")
    parser.add_argument("--output-dir", "--output_dir", required=True, help="Directory for audio, manifests, labels, logs, result.")
    parser.add_argument("--num-gpus", "--num_gpus", type=int, required=True, help="Number of one-GPU worker jobs to launch.")
    parser.add_argument(
        "--sbatch-cmd",
        "--sbatch_cmd",
        required=True,
        help="Sbatch command prefix, e.g. 'sbatch --wait --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=5 --mem=30GB'.",
    )
    parser.add_argument("--audio-root", "--audio_root", default=None, help="Resolve relative audio paths against this dir. Default: input JSON dir.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used inside Slurm workers.")
    parser.add_argument("--worker-setup", "--worker_setup", default="", help="Optional shell snippet inserted before the worker command.")

    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--language", default=None)
    parser.add_argument("--context", default="")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", "--gpu_memory_utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", "--max_model_len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", "--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm-attention-backend", "--vllm_attention_backend", default="FLASH_ATTN", help="Set VLLM_ATTENTION_BACKEND in workers before vLLM import. Use 'auto' to leave it unset.")

    parser.add_argument("--sample-rate", "--sample_rate", type=int, default=16000)
    parser.add_argument("--gap-ms", "--gap_ms", type=float, default=0.0, help="Optional silence inserted between concatenated clips.")
    parser.add_argument("--max-audio-workers", "--max_audio_workers", type=int, default=8, help="Maximum threads used for audio concatenation during preparation. Use 1 for serial preparation.")
    parser.add_argument("--progress-interval", "--progress_interval", type=int, default=100, help="Print preparation progress after this many completed records. Use 0 to disable periodic progress logs.")
    parser.add_argument("--text-separator", "--text_separator", default=" ")
    parser.add_argument("--id-key", "--id_key", default="id")
    parser.add_argument("--output-key", "--output_key", default="output")
    parser.add_argument("--text-key", "--text_key", default="text")
    parser.add_argument("--candidate-key", "--candidate_key", default="candidate_audio_path")
    parser.add_argument("--wer-normalize", "--wer_normalize", choices=("basic", "none"), default="basic")
    parser.add_argument("--wer-tokenizer", "--wer_tokenizer", choices=("word", "char"), default="word")

    parser.add_argument("--result-name", "--result_name", default="result.json")
    parser.add_argument("--overwrite-audio", "--overwrite_audio", action="store_true")
    parser.add_argument("--overwrite-labels", "--overwrite_labels", action="store_true")
    parser.add_argument("--prepare-only", "--prepare_only", action="store_true", help="Prepare audio/manifests but do not submit jobs or collect.")
    parser.add_argument("--collect-only", "--collect_only", action="store_true", help="Skip preparation/submission and collect existing worker outputs.")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Prepare job scripts but do not run sbatch.")
    parser.add_argument("--allow-no-wait", "--allow_no_wait", action="store_true", help="Allow sbatch command without --wait.")
    return parser.parse_args()


def expand_input_paths(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        matches = [Path(p) for p in sorted(glob.glob(value))]
        if not matches:
            matches = [Path(value)]
        for path in matches:
            path = path.expanduser()
            if path.is_dir():
                for suffix in ("*.json", "*.jsonl"):
                    paths.extend(sorted(path.glob(suffix)))
            else:
                paths.append(path)
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def iter_source_records(input_paths: Sequence[Path]) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(str(path))
        for record in iter_json_or_jsonl(path):
            yield path, record


def candidate_count_for_outputs(outputs: Sequence[Dict[str, Any]], candidate_key: str) -> int:
    counts = []
    for idx, item in enumerate(outputs):
        candidates = item.get(candidate_key)
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"output[{idx}].{candidate_key} must be a non-empty list")
        counts.append(len(candidates))
    first = counts[0]
    for idx, count in enumerate(counts):
        if count != first:
            raise ValueError(f"candidate count mismatch: output[0] has {first}, output[{idx}] has {count}")
    return first


def log_main(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [main] {message}", flush=True)


def prepare_one_record(
    source_json: Path,
    record: Dict[str, Any],
    record_index: int,
    args: argparse.Namespace,
    audio_root: Optional[Path],
    prepared_dir: Path,
) -> Dict[str, Any]:
    raw_id = record.get(args.id_key, f"record_{record_index:09d}")
    record_id = str(raw_id)
    try:
        outputs = record.get(args.output_key)
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"{args.output_key} must be a non-empty list")
        for idx, item in enumerate(outputs):
            if not isinstance(item, dict):
                raise ValueError(f"{args.output_key}[{idx}] must be an object")

        n_candidates = candidate_count_for_outputs(outputs, args.candidate_key)
        texts = [str(item.get(args.text_key, "") or "").strip() for item in outputs]
        reference_text = args.text_separator.join(text for text in texts if text)
        if not reference_text:
            raise ValueError("concatenated reference text is empty")

        safe_record = safe_name(record_id)
        group_candidates = []
        for candidate_index in range(n_candidates):
            raw_audio_paths = [str(item[args.candidate_key][candidate_index]) for item in outputs]
            audio_paths = [resolve_audio_path(path, source_json, audio_root) for path in raw_audio_paths]
            concat_path = prepared_dir / safe_record / f"candidate_{candidate_index:03d}.wav"
            if args.overwrite_audio or not concat_path.exists():
                concat_audio_files(audio_paths, concat_path, target_sr=args.sample_rate, gap_ms=args.gap_ms)

            group_candidates.append(
                {
                    "id": record_id,
                    "candidate_index": candidate_index,
                    "candidate": str(candidate_index),
                    "audio": str(concat_path.resolve()),
                    "text": reference_text,
                    "source_json": str(source_json),
                    "num_segments": len(outputs),
                }
            )

        return {
            "ok": True,
            "group": {
                "id": record_id,
                "text": reference_text,
                "candidate_count": n_candidates,
                "candidates": [
                    {
                        "candidate_index": item["candidate_index"],
                        "candidate": item["candidate"],
                        "audio": item["audio"],
                    }
                    for item in group_candidates
                ],
            },
            "candidates": group_candidates,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "id": record_id,
                "source_json": str(source_json),
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def prepare_manifests(args: argparse.Namespace, output_dir: Path) -> Tuple[int, int, int]:
    audio_root = Path(args.audio_root).expanduser().resolve() if args.audio_root else None
    input_paths = expand_input_paths(args.input_json)
    if not input_paths:
        raise ValueError("No input JSON files found")
    if args.max_audio_workers <= 0:
        raise ValueError("--max-audio-workers must be positive")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be >= 0")

    prepared_dir = ensure_dir(output_dir / "prepared_audio")
    manifest_dir = ensure_dir(output_dir / "manifests")
    errors_path = output_dir / "errors.jsonl"
    all_candidates_path = manifest_dir / "all_candidates.jsonl"
    groups_path = manifest_dir / "groups.jsonl"

    record_count = 0
    candidate_count = 0
    error_count = 0
    submitted_count = 0
    completed_count = 0
    start_time = time.time()
    last_log_completed = 0

    def maybe_log_progress(force: bool = False) -> None:
        nonlocal last_log_completed
        if not force and args.progress_interval == 0:
            return
        if not force and completed_count - last_log_completed < args.progress_interval:
            return
        elapsed = max(0.001, time.time() - start_time)
        rate = completed_count / elapsed
        log_main(
            "prepare progress "
            f"submitted={submitted_count} completed={completed_count} "
            f"ok={record_count} errors={error_count} candidates={candidate_count} "
            f"pending={submitted_count - completed_count} "
            f"elapsed_sec={elapsed:.1f} records_per_sec={rate:.2f}"
        )
        last_log_completed = completed_count

    log_main(
        "prepare start "
        f"input_files={len(input_paths)} output_dir={output_dir} "
        f"audio_workers={args.max_audio_workers} sample_rate={args.sample_rate}"
    )

    with all_candidates_path.open("w", encoding="utf-8") as all_f, groups_path.open("w", encoding="utf-8") as groups_f, errors_path.open("w", encoding="utf-8") as err_f:
        def write_prepared_result(result: Dict[str, Any]) -> None:
            nonlocal completed_count, record_count, candidate_count, error_count
            completed_count += 1
            if not result["ok"]:
                error_count += 1
                append_jsonl_record(err_f, result["error"])
                maybe_log_progress()
                return

            for candidate_record in result["candidates"]:
                append_jsonl_record(all_f, candidate_record)
                candidate_count += 1
            append_jsonl_record(groups_f, result["group"])
            record_count += 1
            maybe_log_progress()

        max_workers = max(1, args.max_audio_workers)
        max_pending = max_workers * 4
        pending = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for record_index, (source_json, record) in enumerate(iter_source_records(input_paths)):
                submitted_count += 1
                pending.add(executor.submit(prepare_one_record, source_json, record, record_index, args, audio_root, prepared_dir))
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        write_prepared_result(future.result())

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    write_prepared_result(future.result())

    maybe_log_progress(force=True)
    shard_count = write_shards(all_candidates_path, manifest_dir, max(1, min(args.num_gpus, candidate_count)))
    log_main(
        f"prepare done records={record_count} candidates={candidate_count} "
        f"errors={error_count} shards={shard_count} output_dir={output_dir}"
    )
    return record_count, candidate_count, shard_count


def existing_preparation_summary(output_dir: Path) -> Optional[Tuple[int, int, int]]:
    manifest_dir = output_dir / "manifests"
    all_candidates_path = manifest_dir / "all_candidates.jsonl"
    groups_path = manifest_dir / "groups.jsonl"
    shard_paths = sorted(manifest_dir.glob("shard_*.jsonl"))
    if not all_candidates_path.exists() or not groups_path.exists() or not shard_paths:
        return None

    candidate_count = sum(1 for _ in load_jsonl(all_candidates_path))
    record_count = sum(1 for _ in load_jsonl(groups_path))
    shard_count = sum(1 for path in shard_paths if path.exists() and path.stat().st_size > 0)
    if candidate_count <= 0 or record_count <= 0 or shard_count <= 0:
        return None
    return record_count, candidate_count, shard_count


def write_shards(all_candidates_path: Path, manifest_dir: Path, num_shards: int) -> int:
    shard_paths = [manifest_dir / f"shard_{idx:05d}.jsonl" for idx in range(num_shards)]
    shard_files = [path.open("w", encoding="utf-8") for path in shard_paths]
    counts = [0 for _ in range(num_shards)]
    try:
        for idx, item in enumerate(load_jsonl(all_candidates_path)):
            shard_idx = idx % num_shards
            shard_files[shard_idx].write(json_dumps(item) + "\n")
            counts[shard_idx] += 1
    finally:
        for f in shard_files:
            f.close()

    for path, count in zip(shard_paths, counts):
        if count == 0 and path.exists():
            path.unlink()
    return sum(1 for count in counts if count)


def worker_command(args: argparse.Namespace, worker_path: Path, shard_path: Path, output_path: Path) -> List[str]:
    cmd = [
        args.python,
        str(worker_path),
        "--manifest",
        str(shard_path),
        "--output",
        str(output_path),
        "--model",
        args.model,
        "--backend",
        args.backend,
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--vllm-attention-backend",
        args.vllm_attention_backend,
    ]
    if args.language:
        cmd.extend(["--language", args.language])
    if args.context:
        cmd.extend(["--context", args.context])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.overwrite_labels:
        cmd.append("--overwrite")
    return cmd


def successful_label_keys(output_path: Path) -> set[Tuple[str, int]]:
    done: set[Tuple[str, int]] = set()
    if not output_path.exists():
        return done
    for item in load_jsonl(output_path):
        if item.get("error") is None and item.get("id") is not None and item.get("candidate_index") is not None:
            done.add((str(item["id"]), int(item["candidate_index"])))
    return done


def shard_label_status(shard_path: Path, output_path: Path) -> Tuple[bool, int, int]:
    required = {
        (str(item["id"]), int(item["candidate_index"]))
        for item in load_jsonl(shard_path)
    }
    if not required:
        return True, 0, 0
    done = successful_label_keys(output_path)
    matched_done = len(required.intersection(done))
    return matched_done == len(required), len(required), matched_done


def write_job_scripts(args: argparse.Namespace, output_dir: Path) -> List[Tuple[int, Path, Path]]:
    base_dir = Path(__file__).resolve().parent
    worker_path = base_dir / "worker_qwen3_asr_vllm.py"
    manifest_dir = output_dir / "manifests"
    labels_dir = ensure_dir(output_dir / "worker_outputs")
    slurm_dir = ensure_dir(output_dir / "slurm")
    logs_dir = ensure_dir(output_dir / "logs")

    jobs: List[Tuple[int, Path, Path]] = []
    for shard_path in sorted(manifest_dir.glob("shard_*.jsonl")):
        shard_id = int(shard_path.stem.split("_")[-1])
        output_path = labels_dir / f"{shard_path.stem}.labels.jsonl"
        if not args.overwrite_labels:
            is_complete, total_labels, done_labels = shard_label_status(shard_path, output_path)
            if is_complete:
                log_main(f"resume skip completed shard={shard_id:05d} labels={done_labels}/{total_labels}")
                continue
            if done_labels:
                log_main(f"resume partial shard={shard_id:05d} labels={done_labels}/{total_labels}; worker will process remaining")

        command = " ".join(shlex.quote(part) for part in worker_command(args, worker_path, shard_path, output_path))
        script_path = slurm_dir / f"{shard_path.stem}.sbatch.sh"
        setup = args.worker_setup.strip()
        setup_block = f"{setup}\n" if setup else ""
        script_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f"#SBATCH --job-name=qwen3_label_{shard_id:05d}",
                    f"#SBATCH --output={logs_dir / (shard_path.stem + '.%j.out')}",
                    f"#SBATCH --error={logs_dir / (shard_path.stem + '.%j.err')}",
                    "set -euo pipefail",
                    f"cd {shlex.quote(os.getcwd())}",
                    "export PYTHONUNBUFFERED=1",
                    setup_block + command,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        jobs.append((shard_id, script_path, output_path))
    return jobs


def submit_jobs(args: argparse.Namespace, output_dir: Path) -> None:
    if "--wait" not in shlex.split(args.sbatch_cmd) and not args.allow_no_wait:
        raise ValueError("Collection requires sbatch --wait. Add --wait to --sbatch-cmd or pass --allow-no-wait.")

    jobs = write_job_scripts(args, output_dir)
    if not jobs:
        log_main("no slurm jobs to submit; all label shards are complete")
        return
    log_main(f"submitting slurm_jobs={len(jobs)}")

    submit_log_dir = ensure_dir(output_dir / "sbatch_submit_logs")
    sbatch_prefix = shlex.split(args.sbatch_cmd)
    processes = []
    for shard_id, script_path, _ in jobs:
        cmd = sbatch_prefix + [str(script_path)]
        if args.dry_run:
            log_main("[dry-run] " + " ".join(shlex.quote(part) for part in cmd))
            continue
        log_main(f"submit shard={shard_id:05d} script={script_path}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append((shard_id, cmd, proc))

    failed = []
    for shard_id, cmd, proc in processes:
        stdout, stderr = proc.communicate()
        (submit_log_dir / f"shard_{shard_id:05d}.stdout").write_text(stdout or "", encoding="utf-8")
        (submit_log_dir / f"shard_{shard_id:05d}.stderr").write_text(stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            failed.append((shard_id, proc.returncode, cmd))
        else:
            log_main(f"slurm job done shard={shard_id:05d}")

    if failed:
        for shard_id, returncode, cmd in failed:
            print(f"[main][ERROR] shard={shard_id:05d} returncode={returncode} cmd={' '.join(cmd)}", file=sys.stderr)
        raise RuntimeError(f"{len(failed)} Slurm jobs failed")


def collect_results(args: argparse.Namespace, output_dir: Path) -> Path:
    groups_path = output_dir / "manifests" / "groups.jsonl"
    labels_dir = output_dir / "worker_outputs"
    result_path = output_dir / args.result_name
    if not groups_path.exists():
        raise FileNotFoundError(str(groups_path))

    labels: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for label_path in sorted(labels_dir.glob("*.labels.jsonl")):
        for item in load_jsonl(label_path):
            labels[(str(item.get("id")), int(item.get("candidate_index")))] = item

    results: List[Dict[str, Any]] = []
    missing = 0
    for group in load_jsonl(groups_path):
        record_id = str(group["id"])
        reference_text = str(group["text"])
        detail = []
        best_candidate: Optional[int] = None
        best_wer: Optional[float] = None

        for candidate in group["candidates"]:
            candidate_index = int(candidate["candidate_index"])
            label = labels.get((record_id, candidate_index))
            if label is None:
                missing += 1
                item = {
                    "candidate": candidate_index,
                    "wer": None,
                    "transcript": "",
                    "audio": candidate.get("audio"),
                    "error": "missing worker output",
                }
                detail.append(item)
                continue

            error = label.get("error")
            transcript = str(label.get("asr_text", "") or "")
            candidate_wer = None if error else wer(
                reference_text,
                transcript,
                normalize=args.wer_normalize,
                tokenizer=args.wer_tokenizer,
            )
            item = {
                "candidate": candidate_index,
                "wer": candidate_wer,
                "transcript": transcript,
                "audio": label.get("audio") or candidate.get("audio"),
            }
            if error:
                item["error"] = error
            detail.append(item)

            if candidate_wer is not None and (best_wer is None or candidate_wer < best_wer):
                best_wer = candidate_wer
                best_candidate = candidate_index

        results.append(
            {
                "id": record_id,
                "text": reference_text,
                "best_candidate": best_candidate,
                "detail": detail,
            }
        )

    ensure_dir(result_path.parent)
    result_path.write_text(json_dumps({"results": results, "missing_labels": missing}) + "\n", encoding="utf-8")
    log_main(f"wrote {result_path} groups={len(results)} missing_labels={missing}")
    return result_path


def main() -> int:
    args = parse_args()
    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive")
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    log_main(f"start output_dir={output_dir} num_gpus={args.num_gpus} backend={args.backend}")

    if not args.collect_only:
        existing = None if args.overwrite_audio else existing_preparation_summary(output_dir)
        if existing is not None:
            record_count, candidate_count, shard_count = existing
            log_main(
                "resume skip preparation "
                f"records={record_count} candidates={candidate_count} shards={shard_count} "
                f"reason=existing_manifests use --overwrite-audio to rebuild"
            )
        else:
            _, candidate_count, _ = prepare_manifests(args, output_dir)
        if candidate_count == 0:
            raise RuntimeError("No valid candidate audio was prepared; inspect errors.jsonl")
        if args.prepare_only:
            return 0
        submit_jobs(args, output_dir)
        if args.dry_run:
            return 0

    collect_results(args, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
