# Evaluation guide

## Local setup

The formatter has no required third-party runtime dependencies. Full model evaluation requires Linux, a suitable GPU, local model weights, benchmark prompts, and an inference environment.

The manuscript reports Ubuntu 22.04, Python 3.12, PyTorch 2.12.1, CUDA 13.0, vLLM, and an NVIDIA A100 80GB. Install a compatible environment for your hardware; this release does not claim to have recreated that GPU environment.

Required evaluation packages include vllm, transformers, math-verify, sympy, and numpy. Install additional mathematical packages used by generated code, such as scipy, pandas, mpmath, and matplotlib. The upstream parser may have its own dependencies.

Provide a local `Github.CoRT` checkout containing `infer/parser.py` through `CORT_ROOT`. No upstream source, model weights, or benchmark data are redistributed here.

```bash
export CORT_ROOT=/path/to/local/CoRT
python -m pip install -e .
```

## Input format

Place the five files `test_aime24.reason_step.jsonl`, `test_aime25.reason_step.jsonl`, `test_amc23.reason_step.jsonl`, `test_math500.reason_step.jsonl`, and `test_olympiad_bench.reason_step.jsonl` in a local input directory.

Each line must contain id, problem, prompt, and gt_answer:

```json
{"id":"example-0","problem":"Compute 17 + 4.","prompt":"Compute 17 + 4 using Python if useful.","gt_answer":"21"}
```

The example is a schema illustration. Use the original benchmark prompts for evaluation. The reference answer is used only by the grader, never by the feedback formatter.

## Paper protocol launcher

Preview the commands without running inference:

```bash
python reproduction/run_paper.py --model /path/to/checkpoint --data /path/to/data --output runs/main-1.5b --dry-run
```

Run the three-seed main evaluation:

```bash
python reproduction/run_paper.py --model /path/to/checkpoint --data /path/to/data --output runs/main-1.5b
```

Use a separate output directory and the corresponding checkpoint for the 32B experiment. Run component ablations with:

```bash
python reproduction/run_paper.py --kind ablation --model /path/to/1.5b-checkpoint --data /path/to/data --output runs/ablation-1.5b
```

The launcher supplies seeds 42, 985, and 211; samples 16 for AIME/AMC and four for MATH500/OlympiadBench; 32,768 tokens; and 15 interpreter calls. It records per-seed reports and an aggregate across seeds. The ablation arms are full, no_latest, no_delta, and values_only.

Runners refuse to resume into an existing output file with a different saved configuration. Use a new output directory when changing code or settings. Model outputs may differ across hardware and batching arrangements.

## Scope of reproduction

Repository result tables are transcribed from the manuscript, not recomputed by the release tests. The unit and mocked runner tests verify code behavior, not model accuracy. See [implementation notes](implementation.md) for remaining differences between the available source and the manuscript's pseudocode/grader.
