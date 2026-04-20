# TrackGNN MiniAOD → NanoAOD Porting Plan

This document captures a concrete implementation plan for porting the TrackGNN inference pipeline in `modrundata/DemoAnalyzer.cc` to NanoAOD production in this repository.

## 1) What exists today

- `modrundata/DemoAnalyzer.cc` is an analyzer-style workflow that consumes MiniAOD objects including:
  - `packedPFCandidates` (`tracks`),
  - `lostTracks`,
  - primary vertices,
  - secondary vertices,
  - jets,
  - beamspot,
  - pileup/gen products.
- Inference and graph-building are currently coupled to this MiniAOD-level object access.

Implication: to run at NanoAOD production time, inference must happen **before** event content is flattened and dropped, i.e. during the Nano production step while MiniAOD products are still in event memory.

## 2) Recommended architecture

Use a two-stage extension to Nano production:

1. **EDProducer stage** (`TrackGNNNanoProducer`) running in `nanoSequence`:
   - Consumes MiniAOD products (tracks, lostTracks, vertices, jets, beamspot).
   - Builds graph edges and runs ONNX inference.
   - Produces compact event products:
     - per-edge `src` index vector (`std::vector<uint16_t>` or `uint32_t`),
     - per-edge `dst` index vector,
     - per-edge score vectors for each model output,
     - optional per-jet offsets/counts to map edges back to jets.

2. **FlatTable stage** (`SimpleFlatTableProducer` instances):
   - Exposes the producer outputs as Nano branches.
   - Writes branches into NanoAOD so downstream analysis can read model outputs and `edge_index` directly.

## 3) Data model for saved outputs

Save an event-level edge table with aligned arrays:

- `TrackEdge_src` (`UInt16`/`UInt32`)
- `TrackEdge_dst` (`UInt16`/`UInt32`)
- `TrackEdge_score_<name>` (one float branch per model score)
- `nTrackEdge` (implicit via table length)

Optional jet mapping (if edges are built per jet and you need reverse lookup):

- `Jet_trackEdgeStart` (index into edge table)
- `Jet_trackEdgeCount` (number of edges for that jet)

Why this format:

- Mirrors COO `edge_index` without nested vectors.
- Nano-friendly (flat and columnar).
- Easy reconstruction in analysis:
  - `edge_index = np.stack([src, dst], axis=0)`

## 4) Integration points in this repo

Even though generated `mc_*/data_*_NANO.py` files are not checked in here, this repo already controls Nano config generation and CRAB submission. Integrate via:

- `generateConfigs.sh`: inject one customization import/call for TrackGNN, similar to existing post-generation `sed` customizations.
- New customization module in your CMSSW area (recommended path):
  - `PhysicsTools/NanoAOD/python/trackGNN_cff.py`
  - Expose `def nanoAOD_addTrackGNN(process): ...`

Then append in generated cfgs:

```python
from PhysicsTools.NanoAOD.trackGNN_cff import nanoAOD_addTrackGNN
process = nanoAOD_addTrackGNN(process)
```

## 5) Producer contract (implementation checklist)

When implementing `TrackGNNNanoProducer`:

1. Reuse feature-building logic from `DemoAnalyzer` where possible (extract helper code into shared utility if needed).
2. Keep ONNXRuntime in global cache (thread-safe pattern already used in `DemoAnalyzer`).
3. Enforce deterministic edge ordering (e.g. sorted by `(jetIdx, src, dst)`) so edge-score alignment is stable.
4. Add safety caps for edge multiplicity to avoid pathological event-size blowups.
5. Store integer indices using smallest safe type (`uint16_t` up to 65535 track nodes/event).
6. Add a debug mode to dump first N edges and scores for validation.

## 6) Validation strategy

Run three validation layers:

1. **Unit/event-level parity**:
   - For a fixed MiniAOD file, compare old `DemoAnalyzer` outputs vs new producer outputs.
   - Verify identical edge count/order and score vectors within tolerance.

2. **Nano branch integrity**:
   - Confirm branches exist and lengths match:
     - `len(TrackEdge_src) == len(TrackEdge_dst) == len(score_i)`

3. **Physics-level check**:
   - Reproduce one downstream plot using old pipeline vs Nano-only pipeline.

## 7) Minimal rollout plan

1. Implement producer with only:
   - `src`, `dst`, and one score branch.
2. Validate parity and branch integrity.
3. Add remaining model outputs.
4. Add optional jet-edge mapping branches.
5. Turn on in one year/era config and run a small CRAB dryrun.

## 8) Notes on event size

Edge payload can grow quickly. If needed, add one (or more) controls:

- minimum edge score threshold before writing,
- max edges per jet,
- max edges per event with deterministic truncation.

These controls should be configurable in the customization module.
