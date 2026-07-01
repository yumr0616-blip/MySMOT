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
Learnable (Stage-1a/1b, not yet implemented here): KFA semantic slots, the
Fact Selector's slot, and the KFA/Fact -> MLLM projector(s).
Deterministic (implemented for real in this scaffold): motion fact geometric
extraction, event candidate filtering, output assembly.

## Modules

| Module | File | Frozen / Learnable / Deterministic |
|---|---|---|
| Frozen Tracker | `smot/tracker.py` | Frozen (`StubTracker` stands in for a real detector+SAM2 tracker) |
| Motion Fact Extractor | `smot/motion_facts.py` | Deterministic (real) |
| Event Candidate Filter | `smot/event_filter.py` | Heuristic, not learned (real) |
| Fact Selector | `smot/fact_selector.py` | Slot learnable in Stage-1+; `DeterministicFactSelector` is the Stage-0 default |
| Unary KFA | `smot/kfa.py` | Learnable in Stage-1a; `NoOpUnaryKFA` is the Stage-0 default |
| Pairwise KFA | `smot/kfa.py` | Learnable in Stage-1b; `NoOpPairwiseKFA` is the Stage-0 default |
| Projector | `smot/projector.py` | Learnable in Stage-1a/1b; `NoOpProjector` is the Stage-0 default |
| Frozen MLLM | `smot/mllm.py` | Frozen (`MockMLLMAdapter` stands in for a real MLLM) |
| Output Assembler | `smot/output_assembler.py` | Deterministic (real) |
| Pipeline orchestrator | `smot/pipeline.py` | Wires all of the above |

Core data schemas (`Trajectory`, `Fact`, `PairFeature`, and the
instance/interaction/video assertions) live in `smot/types.py`.

## Stages

- **Stage-0** (implemented here): pure deterministic pipeline — no learning
  at all. Motion Fact Extractor + Event Candidate Filter run for real; the
  Tracker and MLLM are stubs/mocks; Fact Selector/KFA/Projector are
  deterministic/no-op placeholders. This is the "打通" bootstrap milestone.
- **Stage-1a** (future): swap in a learnable Unary KFA (appearance/action
  slots + projector + soft-token injection).
- **Stage-1b** (future): add a learnable Pairwise KFA (interaction slot) +
  structured interaction JSON.

`Pipeline`'s constructor takes every learnable/model-backed component as an
optional argument defaulting to its Stage-0 implementation, so upgrading to
Stage-1a/1b (or plugging in a real tracker/MLLM) requires no change to its
call signature.

## Stack

Pure Python, standard library only (`dataclasses`, `typing.Protocol`,
`unittest`) — no `torch`/`transformers` dependency yet. Those are introduced
when real KFA/Projector slots need soft-token injection via `inputs_embeds`
with backprop (Stage-1a/1b); see the `ml` extra in `pyproject.toml`.

## Running

```bash
# Install (editable) so `import smot` works from anywhere in the repo
pip install -e .

# Run the test suite
python -m unittest discover -s tests -v

# Run the Stage-0 end-to-end demo on a synthetic two-object fixture
python examples/run_stage0.py
```
