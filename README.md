# TDL Challenge 2026 — TopoBench backbones

My contributions to the [TDL Challenge 2026](https://github.com/geometric-intelligence/TopoBench)
(TopoBench). Each folder mirrors the TopoBench directory layout, so its files drop straight into a
TopoBench checkout.

| folder | track | model | paper |
|---|---|---|---|
| [`ordered_gnn/`](ordered_gnn) | Track 1 (GNN) | Ordered GNN — ordered message passing against heterophily and over-smoothing | Song et al., ICLR 2023 (arXiv:2302.01524) |
| [`pcn/`](pcn) | Track 2 (TNN) | Path Complex Network, plus a new **graph → path complex** lifting | Truong & Chin, AAAI 2024 (arXiv:2308.06838) |
| [`polynormer/`](polynormer) | Track 1 (GNN) | Polynormer — polynomial-expressive graph transformer in linear time | Deng et al., ICLR 2024 (arXiv:2403.01232) |

Each folder contains the backbone (`topobench/...`), its Hydra config (`configs/...`), and unit
tests (`test/...`). `PR_DESCRIPTION.md` gives the model summary and design notes written for the
challenge submission.

## What's in each folder

**`ordered_gnn/`**
- `topobench/nn/backbones/graph/ordered_gnn.py`: the backbone, a cumulative-product (ON-LSTM-style)
  gated combine step that aligns hop depth with ordered neuron blocks
- `2026_tdl_challenge/outputs/ordered_gnn/results.json`: the full 72-run challenge grid on
  GraphUniverse (community detection + triangle counting × 3 homophily × 2 degree × 2 power-law
  settings × 3 train seeds)

**`pcn/`**: adds the *path complex* as a new topological domain in TopoBench
- `topobench/transforms/liftings/graph2path_complex.py`: `PathComplexLifting` enumerates simple
  paths up to `max_path_length` and builds boundary maps
- `topobench/nn/backbones/cell/path_complex_network.py`: PCN message passing (PWL test, §4)
- `topobench/nn/wrappers/cell/pcn_wrapper.py`: the TopoBench wrapper
- `2026_tdl_challenge/run_chunk.py`, `merge_chunks.py`: split the evaluation grid into chunks
  and merge them (for time-limited GPU sessions)
- `kaggle/pcn_kaggle.ipynb`: runner notebook for Kaggle GPUs

**`polynormer/`**
- `topobench/nn/backbones/graph/polynormer.py`: local equivariant attention + global linear
  attention (kernel trick, `O(n d²)`)
- `polynormer_gpu_benchmark.ipynb`, `RUN_ON_GPU.md`: a self-contained Kaggle/Colab notebook that
  clones TopoBench, adds the backbone, and runs the full grid

## Using it with TopoBench

```bash
git clone https://github.com/geometric-intelligence/TopoBench.git
cd TopoBench
git checkout 6d8953e                      # base for ordered_gnn/ and pcn/ (polynormer/: 100289d)

cp -r ../tdl-challenge-2026/pcn/{topobench,configs,test} .
git apply ../tdl-challenge-2026/pcn/topobench_registration.patch   # registers the new modules
python -m pytest test/nn/backbones/cell/test_pcn.py test/transforms/liftings/test_graph2path_complex.py
```

`topobench_registration.patch` holds the small edits to existing TopoBench files. Both patches
register dynamically loaded backbone modules in `sys.modules`, which torch_geometric's
`MessagePassing` inspector needs to resolve type hints. The PCN patch also exports the new lifting
and adds it to the in-channel resolver. Polynormer needs no patch; TopoBench's automatic backbone
discovery picks it up.
