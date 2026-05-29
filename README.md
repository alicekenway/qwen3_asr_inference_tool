# Qwen3 vLLM huge-data candidate labeling

This directory contains a Slurm-oriented labeling pipeline:

- `main_labeling.py`: prepares concatenated 16 kHz candidate audio, shards work, submits one-GPU Slurm workers, collects labels, computes WER, and chooses the best candidate.
- `worker_qwen3_asr_vllm.py`: independent Qwen3-ASR/vLLM worker that reads a JSONL manifest and writes JSONL transcripts.
- `common.py`: shared JSON, audio, text-normalization, and WER utilities.

## Input assumption

Each input record has an `id` and an `output` list. Each `output[i]` has:

- `text`: reference text segment.
- `candidate_audio_path`: list of candidate audio paths for the same segment.

For one record, the main program creates:

- Reference text: `output[0].text + " " + output[1].text + ...`
- Candidate audio `0`: concat `output[0].candidate_audio_path[0]`, `output[1].candidate_audio_path[0]`, ...
- Candidate audio `1`: concat `output[0].candidate_audio_path[1]`, `output[1].candidate_audio_path[1]`, ...

All concatenated candidate WAV files are written as 16 kHz mono PCM WAV under:

```text
<output-dir>/prepared_audio/
```

Relative audio paths are resolved against the input JSON file directory by default. Use `--audio-root` if they should be resolved from another root.

## Main Slurm run

```bash
python qwen3_asr_inference_tool/main_labeling.py \
  --input-json '/path/to/input/*.json' \
  --output-dir /path/to/labeling_output \
  --num-gpus 8 \
  --sbatch_cmd 'sbatch --wait --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=5 --mem=30GB' \
  --model Qwen/Qwen3-ASR-1.7B \
  --backend vllm \
  --batch-size 8 \
  --max-new-tokens 1024 \
  --max-audio-workers 10 \
  --progress-interval 100 \
  --gpu-memory-utilization 0.70
```

`--sbatch_cmd` should include `--wait`; the main program starts all `sbatch --wait` processes concurrently, waits for all workers to finish, then collects results.

## Useful options

```bash
# If candidate_audio_path values are relative to a shared data root:
--audio-root /path/to/data/root

# Force a language if wanted:
--language English

# Add optional domain context:
--context 'The audio may contain ASR, TTS, LLM, Qwen, Whisper, NeMo, CTC, Conformer, RNN-T, vLLM.'

# Insert silence between concatenated clips:
--gap-ms 200

# Prepare manifests/audio only:
--prepare-only

# Recollect after workers are already finished:
--collect-only
```

## Main program argument reference

Required orchestration arguments:

- `--input-json`: One or more input JSON/JSONL files, directories, or shell globs. Directories are scanned for `*.json` and `*.jsonl` files. Each record must contain an ID and an output list unless you override the key names below.
- `--output-dir`: Root directory for everything produced by the pipeline. The program writes prepared audio, manifests, Slurm scripts, logs, worker labels, errors, and final results here.
- `--num-gpus`: Number of one-GPU worker jobs to launch. The main program shards candidate audio across this many workers, capped by the number of prepared candidates.
- `--sbatch-cmd`: Sbatch command prefix used to submit each worker. It should request exactly one GPU per job. Include `--wait` so the main process can collect results after all workers finish.

Path and environment arguments:

- `--audio-root`: Optional root directory for resolving relative `candidate_audio_path` values. If omitted, relative paths are resolved relative to the input JSON file that contained the record.
- `--python`: Python executable used inside each Slurm worker script. Defaults to the Python running the main program. Set this if your Slurm environment needs a specific conda/venv Python.
- `--worker-setup`: Optional shell snippet inserted before the worker command in every Slurm script. Use this for commands like `source ~/.bashrc`, `conda activate ...`, `module load ...`, or environment variables.

Qwen3-ASR/vLLM arguments passed to workers:

- `--model`: Model name or path passed to `qwen_asr.Qwen3ASRModel`. Default is `Qwen/Qwen3-ASR-1.7B`.
- `--backend`: ASR backend used by the worker. Default is `vllm`, which calls `Qwen3ASRModel.LLM(...)`. Use `transformers` only when you intentionally want the non-vLLM `from_pretrained(...)` backend.
- `--language`: Optional forced language, for example `English`, `Chinese`, or `Cantonese`. If omitted, the model wrapper can do automatic language handling.
- `--context`: Optional prompt/domain context passed to the ASR model. This is useful when the audio contains domain-specific terms such as model names, acronyms, or product names.
- `--batch-size`: Number of concatenated candidate WAVs processed per ASR batch inside each worker. Increase for throughput if GPU memory allows; decrease if vLLM runs out of memory.
- `--max-new-tokens`: Maximum generated tokens per audio. Larger values are safer for long concatenated audio but use more memory/time.
- `--gpu-memory-utilization`: vLLM GPU memory utilization target. Increase to use more VRAM; decrease if jobs fail from memory pressure.
- `--max-model-len`: Optional vLLM maximum model length. Leave unset unless your environment/model requires explicitly limiting context length.
- `--tensor-parallel-size`: Tensor parallel size passed to the worker. For the intended one-GPU-per-worker setup, keep this as `1`.

Audio preparation arguments:

- `--sample-rate`: Output sample rate for prepared candidate WAVs. Default is `16000`, which matches the requested 16 kHz labeling audio.
- `--gap-ms`: Optional silence inserted between concatenated utterances. Default is `0.0`. Use a small value like `200` if adjacent clips need separation for ASR stability.
- `--max-audio-workers`: Maximum number of threads used by the main program while preparing concatenated candidate audio. Default is `8`. Increase this when audio preparation is I/O-bound and storage can handle more parallel reads/writes; decrease it if the filesystem is overloaded or CPU resampling becomes the bottleneck.
- `--progress-interval`: Print main-program preparation progress after this many completed records. Default is `100`. Use `1` for very detailed logs or `0` to disable periodic progress logs.
- `--text-separator`: String used when concatenating the segment texts into the reference text. Default is a single space.

Input schema arguments:

- `--id-key`: Record ID field name. Default is `id`.
- `--output-key`: Field name containing the list of text/audio candidate elements. Default is `output`.
- `--text-key`: Text field inside each output element. Default is `text`.
- `--candidate-key`: Candidate audio path list field inside each output element. Default is `candidate_audio_path`.

WER and result arguments:

- `--wer-normalize`: Text normalization before WER. `basic` lowercases, Unicode-normalizes, removes punctuation, and collapses spaces. `none` only collapses spaces.
- `--wer-tokenizer`: Tokenization for WER. `word` uses whitespace tokens and falls back to characters for text without spaces. `char` always uses characters.
- `--result-name`: Final result filename under `--output-dir`. Default is `result.json`.

Resume and control arguments:

- `--overwrite-audio`: Recreate prepared concatenated WAV files even if they already exist.
- `--overwrite-labels`: Tell workers to ignore existing successful label records and rewrite worker output labels.
- `--prepare-only`: Stop after preparing audio and manifest shards. No Slurm jobs are submitted.
- `--collect-only`: Skip preparation and Slurm submission, then collect existing worker outputs and recompute the final result.
- `--dry-run`: Write Slurm scripts and print the `sbatch` commands without submitting jobs.
- `--allow-no-wait`: Allow `--sbatch-cmd` without `--wait`. Normally not recommended because collection may run before jobs finish.

## Output directory layout

The pipeline creates these paths under `--output-dir`:

- `prepared_audio/`: Concatenated 16 kHz mono candidate WAVs, grouped by record ID.
- `manifests/all_candidates.jsonl`: Full candidate manifest before sharding.
- `manifests/groups.jsonl`: Per-record grouping metadata used during collection and WER scoring.
- `manifests/shard_*.jsonl`: Worker shard manifests.
- `slurm/shard_*.sbatch.sh`: Generated Slurm scripts.
- `logs/`: Slurm worker stdout/stderr files from `#SBATCH --output` and `#SBATCH --error`.
- `sbatch_submit_logs/`: stdout/stderr from the local `sbatch --wait` submit processes.
- `worker_outputs/*.labels.jsonl`: Raw worker ASR transcripts.
- `errors.jsonl`: Records that failed during preparation, such as missing audio paths or invalid candidate counts.
- `result.json`: Final scored result unless changed by `--result-name`.

## Worker direct run

The worker does not depend on Slurm:

```bash
CUDA_VISIBLE_DEVICES=0 python qwen3_asr_inference_tool/worker_qwen3_asr_vllm.py \
  --manifest /path/to/labeling_output/manifests/shard_00000.jsonl \
  --output /path/to/labeling_output/worker_outputs/shard_00000.labels.jsonl \
  --model Qwen/Qwen3-ASR-1.7B \
  --backend vllm \
  --batch-size 8
```

## Worker argument reference

Required worker arguments:

- `--manifest`: JSONL shard containing prepared candidate audio records. The main program writes these under `manifests/shard_*.jsonl`.
- `--output`: JSONL file where this worker writes ASR labels. The main program expects these under `worker_outputs/*.labels.jsonl`.

Worker model/inference arguments:

- `--model`: Model name or path passed to `qwen_asr.Qwen3ASRModel`.
- `--backend`: `vllm` uses `Qwen3ASRModel.LLM(...)`; `transformers` uses `Qwen3ASRModel.from_pretrained(...)`.
- `--language`: Optional forced language.
- `--context`: Optional domain context prompt.
- `--batch-size`: Number of audio files transcribed per batch.
- `--max-new-tokens`: Maximum generated tokens per audio.
- `--gpu-memory-utilization`: vLLM GPU memory utilization target.
- `--max-model-len`: Optional vLLM maximum model length.
- `--tensor-parallel-size`: Tensor parallel size. Keep this at `1` for one GPU per worker.
- `--overwrite`: Ignore existing successful records in the worker output and rewrite the file from scratch.

## Output

Final output is written to:

```text
<output-dir>/result.json
```

It has this structure:

```json
{
  "results": [
    {
      "id": "record_000001",
      "text": "make it louder the music one",
      "best_candidate": 0,
      "detail": [
        {
          "candidate": 0,
          "wer": 0.0,
          "transcript": "make it louder the music one",
          "audio": "/path/to/prepared_audio/record_000001_xxx/candidate_000.wav"
        }
      ]
    }
  ],
  "missing_labels": 0
}
```

Preparation errors are written to:

```text
<output-dir>/errors.jsonl
```

Worker raw labels are written to:

```text
<output-dir>/worker_outputs/*.labels.jsonl
```

## Dependencies

The worker environment needs the same packages as the notebook demo:

```bash
uv pip install "qwen-asr[vllm]" soundfile tqdm
```

`numpy` is required by audio preparation. `scipy` is optional but preferred for higher-quality resampling. `ijson` is optional and enables streaming for very large JSON array inputs.
