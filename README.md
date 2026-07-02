# MySMOT

Semantic Multi-Object Tracking (SMOT): given a video, a frozen tracker
produces per-object trajectories; on top of that, a frozen multimodal LLM
plus a small set of *learnable* adapter components produce instance-level
behavior captions, pairwise interaction assertions, and a video-level
summary — every claim traceable to a track ID, time span, and evidence
frames.

This repository is not "another intermediate module" — it's a learnable,
compressible, attributable trajectory→language interface.

## Architecture

```
video
  -> [Frozen Tracker] -> trajectories {track_id, box/mask/conf per frame}
  -> three parallel paths into [Frozen MLLM]:
      (1) Motion Transcript       deterministic geometry facts + fact selection
      (2) Unary KFA    -> per-trajectory appearance/action evidence
      (3) Pairwise KFA -> interaction evidence for candidate edges
  -> [Frozen MLLM] (3 prompt types: instance / interaction / video)
  -> structured JSON (attributable) + NL descriptions
```

Frozen (never trained): tracker, MLLM body, MLLM vision tower.
Learnable (Stage-1a implemented in `smot/ml/`): unary KFA slot and the
KFA/Fact -> MLLM projector; Stage-1b adds the pairwise KFA and Fact
Selector slots.
Deterministic (implemented for real): motion fact geometric extraction,
event candidate filtering, output assembly.

## Modules

| Module | File | Frozen / Learnable / Deterministic |
|---|---|---|
| Frozen Tracker | `smot/tracker.py` | Frozen (`StubTracker` stands in for a real detector+SAM2 tracker) |
| Motion Fact Extractor | `smot/motion_facts.py` | Deterministic (real) |
| Event Candidate Filter | `smot/event_filter.py` | Heuristic, not learned (real) |
| Pair Feature builder | `smot/pair_features.py` | Deterministic (real) — per-frame relative geometry fed into the Pairwise KFA |
| Fact Selector | `smot/fact_selector.py` | Slot learnable in Stage-1+; `DeterministicFactSelector` is the Stage-0 default |
| Unary KFA | `smot/kfa.py` | Learnable in Stage-1a; `NoOpUnaryKFA` is the Stage-0 default |
| Pairwise KFA | `smot/kfa.py` | Learnable in Stage-1b; `NoOpPairwiseKFA` is the Stage-0 default |
| Projector | `smot/projector.py` | Learnable in Stage-1a/1b; `NoOpProjector` is the Stage-0 default |
| Frozen MLLM | `smot/mllm.py` | Frozen (`MockMLLMAdapter` stands in for a real MLLM) |
| Output Assembler | `smot/output_assembler.py` | Deterministic (real); structured-JSON-first interaction parsing with free-text fallback |
| Pipeline orchestrator | `smot/pipeline.py` | Wires all of the above |
| Frame features | `smot/frame_features.py` | Deterministic per-frame geometry/motion features — the learnable unary KFA's scoring input |
| BenSMOT converter | `smot/datasets/bensmot.py` | stdlib-only loader: MOT gt.txt -> `Trajectory`, captions/graphml -> gold eval payloads, fact statistics; `probe` CLI for format verification |
| Real MLLM adapter | `smot/ml/qwen_adapter.py` | Frozen Qwen3.5 (`AutoModelForMultimodalLM`), key-frame images + box grounding + soft-token injection via embedding hook |
| Learnable Unary KFA | `smot/ml/unary_kfa.py` | Stage-1a: soft attention readout + hard top-k riding the soft gradient |
| Learnable Projector | `smot/ml/projector.py` | Stage-1a: residual MLP -> m soft tokens, output scale matched to the LM embedding RMS |
| Gradient gate | `smot/ml/gradient_check.py` | Stage-1a acceptance gate #1: non-zero grads land exactly on {unary KFA, projector} |

Core data schemas (`Trajectory`, `Fact`, `PairFeature`, and the
instance/interaction/video assertions) live in `smot/types.py`.

The core package (`smot/` top level and `smot/datasets/`) stays
stdlib-only; everything needing torch/transformers/opencv/PIL lives in
`smot/ml/` and the dependency direction is strictly `smot.ml -> smot`.

## Stages

- **Stage-0** (done): pure deterministic pipeline — no learning at all.
  Motion Fact Extractor + Event Candidate Filter run for real; the Tracker
  and MLLM are stubs/mocks; Fact Selector/KFA/Projector are
  deterministic/no-op placeholders. This is the "打通" bootstrap milestone.
- **Real inference** (done): `QwenMLLMAdapter` replaces the mock — frozen
  Qwen3.5-2B consumes annotated key frames (per-track colored boxes +
  color legend) and answers the interaction task in structured JSON.
  BenSMOT GT trajectories stand in for the frozen tracker.
- **Stage-1a** (done): `LearnableUnaryKFA` + `MLPProjector`, soft tokens
  injected through an embedding forward hook (input_ids keep flowing
  through the model so all internal multimodal fusion stays intact),
  placed at the end of the user turn — before the assistant header.
  `python -m smot.ml.gradient_check` passes: loss backprops through the
  frozen LM into exactly {unary KFA, projector}, 617 frozen tensors get no
  gradient. `python -m smot.ml.training` teacher-forces all three tasks
  through one shared CE loss (the exact same forward path the gate
  verifies), z-scores fact embeds with dataset statistics carried inside
  the checkpoint, and logs a per-step loss curve.
- **Stage-1b** (future): learnable Pairwise KFA + Fact Selector slots.

`Pipeline`'s constructor takes every learnable/model-backed component as an
optional argument defaulting to its Stage-0 implementation, so upgrading to
Stage-1a/1b (or plugging in a real tracker/MLLM) requires no change to its
call signature. The Stage-1 seams are already live in Stage-0: the KFA
protocols accept per-frame visual features / `PairFeature` sequences (unused
by the no-op defaults), and the projector's output is wired into
`MLLMRequest.soft_tokens` — a real projector's tokens reach the MLLM adapter
with no further plumbing.

Every `PipelineResult` carries a `cost` report (`n_vlm_calls`,
`n_key_frames`, `n_facts_selected`, `n_soft_tokens`) so the §7 cost metrics
are measurable from day one.

## Stack

The core package is standard library only (`dataclasses`,
`typing.Protocol`, `unittest`). The `smot/ml/` area needs the `ml` extra;
on RTX 50-series (Blackwell / sm_120) torch must be the cu128 build:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -e .[ml]
# Windows 控制台中文输出乱码时:
$env:PYTHONUTF8 = "1"
```

Qwen3.5-2B weights (~4 GB) download on first use; behind the GFW set
`$env:HF_ENDPOINT = "https://hf-mirror.com"` first.

## Running

```bash
# Stdlib-only: tests (ml tests auto-skip without torch) and the Stage-0 demo
python -m unittest discover -s tests -v
python examples/run_stage0.py

# BenSMOT workflow (download from https://github.com/HengLan/SMOT first):
# 1. verify the annotation-format assumptions on one real sequence
python -m smot.datasets.bensmot probe <BenSMOT>/test/<activity>/<seq>
# 2. deterministic Mock baseline + SS7 eval (cost floor)
python examples/run_bensmot_stage0.py <BenSMOT>/test --limit 20
# 3. dataset-level fact statistics (Stage-1a norm_value normalization)
python -m smot.datasets.bensmot stats <BenSMOT>/train -o fact_stats.json
# 4. real frozen Qwen3.5 end-to-end (needs the ml venv)
.venv/Scripts/python examples/run_bensmot_real.py <BenSMOT>/test --limit 5
#    ... add --checkpoint stage1a.pt to inject trained Stage-1a components

# Stage-1a acceptance gate #1 (needs the ml venv + GPU)
.venv/Scripts/python -m smot.ml.gradient_check

# Stage-1a training (frozen Qwen3.5; trains only unary KFA + projector,
# ~2.3M params; fits the 8 GB RTX 5060). Then eval the checkpoint against
# the M-A2 baseline via run_bensmot_real.py --checkpoint.
.venv/Scripts/python -m smot.ml.training <BenSMOT>/train --limit 200 \
    --out-dir out/stage1a --epochs 2
.venv/Scripts/python examples/run_bensmot_real.py <BenSMOT>/test --limit 5 \
    --checkpoint out/stage1a/stage1a.pt

# Evaluate any pred/gold payload pair (SS7: tiered interaction F1
# strict / synonym-merged / coarse, direction accuracy, instance coverage,
# cost aggregation). Both files hold a PipelineResult JSON or a list of them.
python -m smot.eval pred.json gold.json
```
