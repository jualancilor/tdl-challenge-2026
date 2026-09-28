"""Unit tests for the PCN backbone (arXiv:2308.06838)."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.backbones.cell.path_complex_network import (
    PCN,
    upper_neighborhood_from_incidence,
)
from topobench.transforms.liftings import PathComplexLifting

IN_CHANNELS = 6
HIDDEN_CHANNELS = 16
OUT_CHANNELS = 8


def _lifted_graph(max_path_length=2, seed=0):
    """Lift a small test graph to a path complex.

    The graph is a triangle 0-1-2 with a tail 2-3-4, so that every
    rank of the complex is non-empty and both boundary and upper
    neighborhoods are exercised.

    Parameters
    ----------
    max_path_length : int, optional
        Maximum rank of the lifted path complex.
    seed : int, optional
        Seed for the node features.

    Returns
    -------
    torch_geometric.data.Data
        The lifted data.
    """
    generator = torch.Generator().manual_seed(seed)
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
    edge_index = torch.tensor(
        [[u for u, v in edges] + [v for u, v in edges],
         [v for u, v in edges] + [u for u, v in edges]],
        dtype=torch.long,
    )
    data = Data(
        x=torch.randn(5, IN_CHANNELS, generator=generator),
        edge_index=edge_index,
        num_nodes=5,
        y=torch.tensor([0]),
    )
    lifting = PathComplexLifting(max_path_length=max_path_length)
    return lifting.forward(data)


def _make_model(**kwargs):
    """Instantiate a PCN model with the test channel sizes.

    Parameters
    ----------
    **kwargs : dict
        Overrides for the PCN constructor.

    Returns
    -------
    PCN
        The instantiated model.
    """
    params = {
        "in_channels": IN_CHANNELS,
        "hidden_channels": HIDDEN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "num_layers": 2,
        "max_path_length": 2,
        "dropout": 0.0,
        "readout": "sum",
    }
    params.update(kwargs)
    return PCN(**params)


def test_pcn_output_shape():
    """Test that the forward pass produces correct output shapes."""
    data = _lifted_graph()
    model = _make_model()
    x_0, x_1, x_2 = model(
        data.x_0, data.x_1, data.x_2, data.incidence_1, data.incidence_2
    )
    assert x_0.shape == (data.x_0.shape[0], OUT_CHANNELS)
    assert x_1.shape == (data.x_1.shape[0], OUT_CHANNELS)
    assert x_2.shape == (data.x_2.shape[0], OUT_CHANNELS)


def test_pcn_gradient_flow():
    """Test that backward runs and every parameter receives a grad."""
    data = _lifted_graph()
    model = _make_model(num_layers=3, dropout=0.1)
    x_0, x_1, x_2 = model(
        data.x_0, data.x_1, data.x_2, data.incidence_1, data.incidence_2
    )
    loss = x_0.sum() + x_1.sum() + x_2.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"


def test_pcn_device_consistency():
    """Test that the output device matches the input device (CPU)."""
    data = _lifted_graph()
    model = _make_model()
    outputs = model(
        data.x_0, data.x_1, data.x_2, data.incidence_1, data.incidence_2
    )
    for out in outputs:
        assert out.device == data.x_0.device


def test_pcn_deterministic_eval():
    """Test that eval mode is deterministic despite dropout."""
    data = _lifted_graph()
    model = _make_model(dropout=0.5)
    model.eval()
    with torch.no_grad():
        out_a = model(
            data.x_0,
            data.x_1,
            data.x_2,
            data.incidence_1,
            data.incidence_2,
        )
        out_b = model(
            data.x_0,
            data.x_1,
            data.x_2,
            data.incidence_1,
            data.incidence_2,
        )
    for a, b in zip(out_a, out_b, strict=True):
        assert torch.equal(a, b)


def test_pcn_max_path_length_1():
    """Test the degenerate case where path cells stop at edges."""
    data = _lifted_graph(max_path_length=1)
    model = _make_model(max_path_length=1)
    x_0, x_1, x_2 = model(
        data.x_0, data.x_1, incidence_1=data.incidence_1
    )
    assert x_0.shape == (data.x_0.shape[0], OUT_CHANNELS)
    assert x_1.shape == (data.x_1.shape[0], OUT_CHANNELS)
    assert x_2 is None


def test_pcn_max_path_length_2():
    """Test the standard case with path cells up to length 2."""
    data = _lifted_graph(max_path_length=2)
    model = _make_model(max_path_length=2)
    x_0, x_1, x_2 = model(
        data.x_0, data.x_1, data.x_2, data.incidence_1, data.incidence_2
    )
    assert x_2 is not None
    assert x_2.shape == (data.x_2.shape[0], OUT_CHANNELS)


def test_pcn_readout_variants():
    """Test that 'sum' and 'mean' readouts produce correct shapes."""
    data = _lifted_graph()
    for readout in ["sum", "mean"]:
        model = _make_model(readout=readout)
        outputs = model(
            data.x_0,
            data.x_1,
            data.x_2,
            data.incidence_1,
            data.incidence_2,
        )
        pooled = model.readout_complex(list(outputs))
        assert pooled.shape == (1, OUT_CHANNELS)
        # Per-rank batch assignments of two complexes.
        batch_indices = [
            torch.zeros(x.shape[0], dtype=torch.long) for x in outputs
        ]
        for batch in batch_indices:
            if batch.numel() > 0:
                batch[-1] = 1
        pooled_batched = model.readout_complex(
            list(outputs), batch_indices
        )
        assert pooled_batched.shape == (2, OUT_CHANNELS)
        # Empty ranks are skipped by the readout.
        pooled_partial = model.readout_complex(
            [outputs[0], outputs[1], None]
        )
        assert pooled_partial.shape == (1, OUT_CHANNELS)


def test_pcn_single_layer():
    """Test that a single-layer model runs."""
    data = _lifted_graph()
    model = _make_model(num_layers=1)
    x_0, x_1, x_2 = model(
        data.x_0, data.x_1, data.x_2, data.incidence_1, data.incidence_2
    )
    assert x_0.shape == (data.x_0.shape[0], OUT_CHANNELS)
    assert x_2.shape == (data.x_2.shape[0], OUT_CHANNELS)


def test_pcn_invalid_hyperparameters():
    """Test that invalid hyperparameters raise ValueError."""
    with pytest.raises(ValueError, match="max_path_length"):
        _make_model(max_path_length=3)
    with pytest.raises(ValueError, match="readout"):
        _make_model(readout="max")


def test_pcn_empty_top_rank():
    """Test a complex whose top rank is empty.

    When max_path_length is 2 but the batch contains no 2-paths (or
    the rank-2 inputs are missing altogether), the model must still
    run, with zero upper messages at rank 1.
    """
    data = _lifted_graph()
    model = _make_model()
    x_0, x_1, x_2 = model(
        data.x_0, data.x_1, x_2=None, incidence_1=data.incidence_1
    )
    assert x_0.shape == (data.x_0.shape[0], OUT_CHANNELS)
    assert x_1.shape == (data.x_1.shape[0], OUT_CHANNELS)
    assert x_2.shape == (0, OUT_CHANNELS)


def test_upper_neighborhood_from_incidence():
    """Test the upper neighborhood derivation on a hand-solved case.

    For the triangle, every 2-path is bounded by all 3 edges, so each
    of the 3 cofaces induces 3 * 2 = 6 ordered pairs of upper-adjacent
    edges: 18 pairs in total (Definition 5 of arXiv:2308.06838).
    """
    incidence_2 = torch.ones(3, 3).to_sparse_coo()
    up_index, shared_coface = upper_neighborhood_from_incidence(
        incidence_2
    )
    assert up_index.shape == (2, 18)
    assert shared_coface.shape == (18,)
    # No self-pairs, and the shared coface has both pair members among
    # its faces (trivially true for the triangle; checked exactly).
    assert (up_index[0] != up_index[1]).all()
    assert torch.equal(
        torch.bincount(shared_coface), torch.tensor([6, 6, 6])
    )
    # An empty incidence yields an empty neighborhood.
    empty = torch.sparse_coo_tensor(
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0),
        size=(4, 0),
    )
    up_index, shared_coface = upper_neighborhood_from_incidence(empty)
    assert up_index.shape == (2, 0)
    assert shared_coface.shape == (0,)
