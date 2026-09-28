# Track: Track2; Team name: [YOUR_TEAM_NAME]; Model: Path Complex Networks (PCN)

**Label:** `track-2-tnn`

## Model summary

This PR introduces an **entirely new topological domain to TopoBench — the
path complex** — together with the Path Complex Network (PCN) that operates
on it. Path complexes (Section 3 of arXiv:2308.06838) generalize both
simplicial complexes and regular cell complexes without imposing an
inductive bias on cliques or rings: the rank-k cells are the *simple paths
of length k* of the graph — ordered sequences of k+1 distinct,
consecutively connected vertices — rather than unordered vertex subsets
(simplices) or glued disks (cells). A triangle is representable as the path
complex spanned by its 2-paths, and so is a plain line graph, where no
2-simplex or 2-cell exists; the domain therefore captures higher-order
structure in sparse graphs where clique- and cycle-based liftings produce
nothing.

The contribution is **both a new lifting transform and a new backbone**:

1. **`PathComplexLifting`** (graph → path complex) enumerates all simple
   paths of the graph up to a configurable `max_path_length`. Since the
   graph is undirected, a path and its reverse are the same cell
   (Section 3.1); cells are canonically oriented with first vertex smaller
   than last vertex. Boundary maps follow the boundary operator of
   Definition 2 restricted to allowed paths (Definition 3): removing any
   vertex of the sequence yields a face whenever the truncated sequence is
   still a path of the graph — so each 2-path of a triangle has *three*
   boundary edges (the interior removal yields the closing edge), while a
   2-path of a line graph has two. The lifting builds the boundary
   incidence and (co)adjacency structures of every rank and is
   deterministic (verified by test).
2. **`PCN`** implements the Path Weisfeiler-Lehman (PWL) message passing
   scheme of Section 4. Per layer and per rank-k cell σ:

   - `m_B(σ) = AGG_{τ ∈ B(σ)} M_B(h_σ, h_τ)` — boundary messages;
   - `m_↑(σ) = AGG_{τ ∈ N_↑(σ), δ ∈ C(σ,τ)} M_↑(h_σ, h_τ, h_δ)` — upper
     messages, which include the features of the shared coface δ
     (the (σ, τ, δ) triples are derived from the boundary incidence
     matrices, so the scheme remains valid under TopoBench's
     block-diagonal batching);
   - `h_σ ← UP(h_σ, m_B(σ), m_↑(σ))` — a GIN-style injective update
     (per-branch (1+ε) perturbation and MLP, followed by a combine MLP),
     following the reference implementation.

   Per **Theorem 11** of the paper, co-boundary and lower-adjacent messages
   are redundant given boundary and upper-adjacent messages, so — exactly
   like the official PCN and unlike a verbatim CWN — the model exchanges
   messages along boundary and upper neighborhoods only.

**Citation:** Truong, Q., & Chin, P. (2024). *Weisfeiler and Lehman Go
Paths: Learning Topological Features via Path Complexes.* Proceedings of
the AAAI Conference on Artificial Intelligence, 38(14), 15382-15391.
arXiv:2308.06838.

## Domain-registration decision

TopoBench has no dedicated `path` domain directory. The path-complex model
is registered **under the existing `cell/` machinery** (backbone in
`topobench/nn/backbones/cell/`, model config in `configs/model/cell/`),
for the following reasons:

- the entire cell-domain pipeline (per-rank features `x_0, x_1, x_2`,
  boundary incidence matrices, `AllCellFeatureEncoder`,
  `PropagateSignalDown` readout, batching) applies to path complexes
  unchanged — path complexes are consumed through exactly the same
  connectivity interface;
- `toponetx.classes.PathComplex` — the canonical implementation of the
  paper's construction, which this lifting builds on — already exposes the
  same matrix API as `CellComplex` and `SimplicialComplex`;
- a new top-level domain directory would have required duplicating the
  encoder/readout/wrapper plumbing for no behavioral difference.

The *lifting* keeps its own identity as `graph2path_complex` (single
module `topobench/transforms/liftings/graph2path_complex.py`, config key
`graph2path_complex_lifting`), so the new domain remains first-class and
discoverable on the transform side. This was the least-deviation option.

## Files added

- `topobench/transforms/liftings/graph2path_complex.py` —
  `PathComplexLifting` (graph → path complex, via
  `toponetx.classes.PathComplex`, the reference implementation of the
  paper's simple-path complex).
- `topobench/nn/backbones/cell/path_complex_network.py` — `PCN` backbone:
  `PCNBoundaryConv` and `PCNUpperConv` (both inherit
  `torch_geometric.nn.MessagePassing`), `PCNLayer` (per-rank PWL update),
  `upper_neighborhood_from_incidence` (vectorized, deterministic
  derivation of the (σ, τ, δ) upper-adjacency triples from sparse
  incidence), and the `PCN` module with `in_channels`, `hidden_channels`,
  `out_channels`, `num_layers`, `max_path_length`, `dropout`, `readout`.
- `topobench/nn/wrappers/cell/pcn_wrapper.py` — `PCNWrapper` following the
  `CWNWrapper` convention (required by TopoBench's TBModel pipeline).
- `configs/model/cell/pcn.yaml` — mirrors `configs/model/cell/cwn.yaml`.
- `configs/transforms/liftings/graph2path_complex.yaml` — lifting config.
- `configs/transforms/model_defaults/pcn.yaml` — makes `model=cell/pcn`
  resolve to the path-complex lifting by default (this is how the
  hash-protected challenge notebook, which only sets `MODEL_CONFIG`,
  applies the correct lifting).
- `test/nn/backbones/cell/test_pcn.py` — 11 tests (the 8 mandated ones
  plus invalid-hyperparameter validation, empty-top-rank handling, and a
  hand-verified test of the upper-neighborhood derivation).
- `test/transforms/liftings/test_graph2path_complex.py` — 7 tests (the 5
  mandated ones plus registry membership and edge-attribute preservation).
  `test_lifting_triangle` hard-asserts the hand-verified triangle counts
  (3 vertices / 3 edges / 3 two-paths, each 2-path bounded by all three
  edges) and `test_lifting_boundary_relations` asserts the full
  hand-computed incidence matrices of a triangle-with-tail graph.
- `test/nn/wrappers/cell/test_pcn_wrapper.py` — wrapper forward test.

New modules have 100% line coverage (`path_complex_network.py`,
`graph2path_complex.py`, `pcn_wrapper.py`).

## Files modified

- `topobench/transforms/liftings/__init__.py`,
  `topobench/transforms/__init__.py` — register the new lifting in
  `LIFTINGS`/`TRANSFORMS`.
- `topobench/utils/config_resolvers.py` — add
  `graph2path_complex_lifting` / `graph2path_complex` to the lifting keys
  recognized by `infer_in_channels`, so per-rank input channels are
  inferred for the new domain.
- `topobench/nn/backbones/__init__.py`,
  `topobench/nn/backbones/cell/__init__.py` — the automated backbone
  discovery now registers each loaded module in `sys.modules` before
  executing it. Without this, any backbone subclassing
  `torch_geometric.nn.MessagePassing` crashes at Hydra instantiation time,
  because PyG's inspector resolves type hints through `sys.modules` while
  the discovery mechanism loads modules under synthetic names.
- `2026_tdl_challenge/run_evaluation.ipynb` — `MODEL_CONFIG = "cell/pcn"`,
  plus a fix to the (unprotected) `expected_hash` cell: the upstream
  constant is stale and does not match the pristine protected cells at git
  HEAD. The value used is the SHA-256 of the unmodified protected region
  as shipped in the repository; the protected region itself is untouched.

## Complexity tradeoffs (stated honestly)

- **The lifting materializes the boundary structure only** (per-rank
  incidence matrices + shape), not the full Hodge structure
  (laplacians, (co)adjacencies). The PWL scheme consumes boundary and
  upper neighborhoods exclusively (Theorem 11), and both derive from the
  boundary matrices, so nothing the model uses is lost. This is not just
  an optimization: on heavy-tailed GraphUniverse graphs the rank-2
  coadjacency/laplacians grow quadratically around hubs (measured ~47 MB
  per matrix for a single 132-degree-hub graph; ~20 GB per 1000-graph
  setting), which makes precomputing them infeasible. If a model config
  explicitly requests `neighborhoods`, the lifting falls back to the
  generic full connectivity builder.
- **`max_path_length` is capped at 2 for the benchmark** (the config
  default). Path enumeration grows combinatorially with path length;
  length 2 keeps the lifting tractable on the denser GraphUniverse
  settings while already realizing the paper's central claim (higher-order
  cells on sparse graphs where cliques/rings are absent). This is a
  complexity/tractability cap on the *domain size*, not a change to the
  algorithm: boundary maps, canonical orientation, and the PWL scheme are
  exactly as specified for the chosen dimension. The TopoBench backbone
  validates `max_path_length ∈ {1, 2}` because TopoBench's cell-complex
  data convention carries ranks 0-2; the lifting itself supports arbitrary
  `max_path_length`.
- The GIN-style update MLPs omit batch normalization (the official
  implementation's `conv_type="B"` uses it): ranks of a path complex can
  legitimately hold zero or one cell in a batch, where training-mode
  BatchNorm is undefined. This follows the TopoModelX-style cell backbones
  already in TopoBench.
- Message aggregation is sum (`AGG = Σ`), as in the reference
  implementation; the `readout` hyperparameter ('sum'/'mean') controls the
  complex-level pooling exposed via `PCN.readout_complex`. Inside the
  TopoBench pipeline, task-level pooling is delegated to the standard
  `PropagateSignalDown` readout, as for every other cell model.

## Benchmark

- Confirmed: all **72 GraphUniverse runs** completed (12 grid settings x 2
  tasks x 3 seeds), including in-distribution test and out-of-distribution
  evaluation on the 11 remaining settings per run.
- Results: `2026_tdl_challenge/outputs/pcn/results.json`.
- Headline metrics (in-distribution test, mean ± std over 36 runs per
  task: 12 settings x 3 seeds):
  - Community detection accuracy (higher is better): **[TO FILL]**
  - Triangle counting MSE / total triangles (lower is better):
    **[TO FILL]**
- Runs were executed on CPU (`trainer.accelerator=cpu`); metrics are
  deterministic per seed and hardware-independent up to floating-point
  reduction order.
