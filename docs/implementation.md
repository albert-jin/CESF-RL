# Implementation notes

The core file `cesf/execution_state.py` is copied byte for byte from the available state-v3c implementation. `source_manifest.json` records source and release hashes. The integration API adds trajectory-local state management without changing the core selection or formatting rules.

## Exact source behavior

- AST discovery inspects Assign, AnnAssign, and AugAssign in the latest block. Non-augmented assignments with Constant, Name, List, Tuple, Dict, or Set right-hand sides are excluded.
- Print-mentioned computed names come first, followed by reversed AST discovery order with duplicates removed. This is a static heuristic, not dynamic branch tracking or guaranteed chronological assignment order.
- The six-binding cap is applied after missing, private, and unsupported values are skipped.
- Python numeric values use repr; Fraction and Decimal use str. For example, a Fraction appears as `448/15625`, not `Fraction(448, 15625)`.
- Strings, dictionaries, and sets are not supported value outputs in this extracted core. Short lists and tuples are supported, as are eligible NumPy and symbol-free SymPy values.
- Scalar representations over 80 characters are omitted. The value 2,048 in this source controls whether a container's unique count is attempted; it is not a universal representation-length rejection threshold.
- The delta renderer proposes at most three changed binding lines. Token packing counts the header with the actual tokenizer and skips a non-fitting proposed line, potentially retaining a later shorter line.
- When any feedback is emitted, the runner saves the complete current candidate snapshot. It does not save only the emitted subset. With no emitted feedback, the saved snapshot is unchanged.

These details differ in places from the simplified pseudocode and overview figure in the manuscript. The source has not been silently rewritten to imply that the released code generated a different set of results.

## Evaluation adapter

Evaluation settings and results in the README come exclusively from the current TeX manuscript. The adapted runners accept explicit model paths, seeds, and sample counts. Their tool-call loop permits the manuscript's 15 calls, followed by a final reasoning round. The ablation runner accepts multiple samples per question.

The paper launcher uses seeds 42, 985, and 211, with 16 samples for AIME/AMC and four for MATH500/OlympiadBench. The worker replays accumulated code in a fresh namespace, then extracts candidate values using the latest block. The paired runner forks a shared trajectory at the first actual feedback insertion.

The supplied grader is the extracted whole-generation Math-Verify grader. The manuscript also describes a DeepScaler symbolic fallback; that fallback is not implemented by the extracted runner and remains an integration requirement for exact protocol reproduction.

The workers use Linux resource limits and an AST policy. They are research execution helpers, not a security boundary for arbitrary untrusted programs. Run generated code inside an isolated environment appropriate to your deployment.

## Figure provenance

All five PNGs were copied from the manuscript's figure directory. `paper_pipeline.png` preserves the author-selected diagram, including decorative Python and OpenAI icons. The OpenAI icon is illustrative and does not identify the CoRT backbone as an OpenAI model.

The experimental plots and CSV tables reproduce the manuscript's reported values. No historical development-run values are used in the README or results tables.
