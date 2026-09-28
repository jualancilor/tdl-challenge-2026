# Track: Track1; Team name: [YOUR_TEAM_NAME]; Model: Ordered GNN

**Label:** `track-1-gnn`

## Model summary

Ordered GNN reorganizes the *combine* stage of message passing so that the
nested rooted-tree hierarchy around every node is aligned with ordered
blocks of neurons in its representation: inner neuron blocks are reserved
for ego and near-hop information, while messages arriving at deeper layers
may only write into the outer blocks. The alignment is enforced with a
cumulative-product gating mechanism adapted from the ordered-neuron idea of
ON-LSTM: a master forget gate is passed through `torch.cumprod`, producing
a monotonically non-increasing gate across neuron blocks, so whenever block
*j* is opened for updating every block outside *j* is opened as well. A
shifted input gate admits the newly aggregated context precisely into the
blocks the forget gate has released. Because hop-specific information is
kept in dedicated neuron blocks instead of being averaged into the whole
embedding, ego signal survives deep stacks (no over-smoothing) and far-hop
noise cannot corrupt near-hop features (robustness under heterophily) —
both from a single mechanism, without any heterophily-specific design.

**Citation:** Song, Y., Zhou, C., Wang, X., & Lin, Z. (2023). *Ordered GNN:
Ordering Message Passing to Deal with Heterophily and Over-smoothing.*
International Conference on Learning Representations (ICLR 2023).
arXiv:2302.01524.

## Files added

- `topobench/nn/backbones/graph/ordered_gnn.py` — `OrderedGNNLayer`
  (inherits `torch_geometric.nn.MessagePassing`, configurable `aggr`,
  default `mean`) and `OrderedGNN` (input projection → stacked ordered
  layers with optional residual connections and layer normalization →
  output projection). Includes the paper's chunking trick (Section 2.5)
  via an optional `chunk_size`.
- `configs/model/graph/ordered_gnn.yaml` — Hydra config following the
  structure of `configs/model/graph/gps.yaml`.
- `test/nn/backbones/graph/test_ordered_gnn.py` — 14 self-contained unit
  tests (output shape, gradient flow to every parameter, device
  consistency, eval-mode determinism, no-residual / no-layer-norm
  variants, single-layer and 8-layer stacks, batch-vector acceptance, gate
  monotonicity, chunked gating, validation errors, skip projection,
  alternative aggregations). `ordered_gnn.py` has 100% line coverage.

## Files modified

- `topobench/nn/backbones/__init__.py` and
  `topobench/nn/backbones/graph/__init__.py` — the automated backbone
  discovery now registers each loaded module in `sys.modules` before
  executing it. Without this, any backbone subclassing
  `torch_geometric.nn.MessagePassing` (OrderedGNN is the first in the
  repo) crashes at Hydra instantiation time, because PyG's inspector
  resolves the class's type hints through `sys.modules` and the discovery
  mechanism loads modules under synthetic names. All 164 existing backbone
  tests still pass.

## Benchmark

- Confirmed: all **72 GraphUniverse runs** completed (12 grid settings x 2
  tasks x 3 seeds), including in-distribution test and out-of-distribution
  evaluation on the 11 remaining settings per run.
- Results: `2026_tdl_challenge/outputs/ordered_gnn/results.json`.
- Headline metrics (in-distribution test, mean +/- std over 36 runs per
  task: 12 settings x 3 seeds):
  - Community detection accuracy (higher is better): **0.4407 +/- 0.1440**
    (best setting 0.7014, hardest 0.2660)
  - Triangle counting MSE / total triangles (lower is better):
    **0.7947 +/- 1.0808** (best setting 0.0121)
- Runs were executed on CPU (`trainer.accelerator=cpu`); metrics are
  deterministic per seed and hardware-independent up to floating-point
  reduction order.

## Why cumprod gating addresses heterophily and over-smoothing at once

The cumulative product of per-block forget gates is monotone by
construction, so the model can only *append* new-hop information outward —
it can never overwrite the inner blocks holding ego and near-hop signal.
Under heterophily, dissimilar far-hop neighborhoods are quarantined in
outer blocks instead of being blended into the ego representation; under
deep stacking, the same mechanism prevents representations from collapsing
toward a neighborhood average, since the inner blocks are provably
preserved by gates that multiply to values near 1. One ordering
constraint therefore tackles both failure modes simultaneously.
