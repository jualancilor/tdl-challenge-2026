"""Unit tests for the Ordered GNN backbone (arXiv:2302.01524)."""

import pytest
import torch

from topobench.nn.backbones.graph.ordered_gnn import (
    OrderedGNN,
    OrderedGNNLayer,
)


def test_ordered_gnn_output_shape():
    """Forward pass produces shape (num_nodes, out_channels)."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=2,
    )
    out = model(x, edge_index)

    assert out.shape == (num_nodes, out_channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_ordered_gnn_gradient_flow():
    """loss.backward() runs and all parameters receive gradients."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.25,
    )
    model.train()

    out = model(x, edge_index)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"


def test_ordered_gnn_device_consistency():
    """Output device matches input device (CPU)."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=2,
    )
    out = model(x, edge_index)

    assert out.device == x.device


def test_ordered_gnn_deterministic_eval():
    """Two forward passes in eval() mode produce identical output."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.5,
    )
    model.eval()

    out1 = model(x, edge_index)
    out2 = model(x, edge_index)

    assert torch.allclose(out1, out2)


def test_ordered_gnn_no_residual():
    """residual=False still produces the correct output shape."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=2,
        residual=False,
    )
    out = model(x, edge_index)

    assert model.residual is False
    assert out.shape == (num_nodes, out_channels)


def test_ordered_gnn_no_layer_norm():
    """layer_norm=False still produces the correct output shape."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=2,
        layer_norm=False,
    )
    out = model(x, edge_index)

    assert model.norms is None
    assert out.shape == (num_nodes, out_channels)


def test_ordered_gnn_single_layer():
    """num_layers=1 runs without error."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=1,
    )
    out = model(x, edge_index)

    assert len(model.convs) == 1
    assert out.shape == (num_nodes, out_channels)


def test_ordered_gnn_deep():
    """num_layers=8 runs without error (over-smoothing resilience)."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=8,
    )
    out = model(x, edge_index)

    assert len(model.convs) == 8
    assert out.shape == (num_nodes, out_channels)
    assert not torch.isnan(out).any()


def test_ordered_gnn_batch_vector():
    """Passing a dummy batch tensor does not raise."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=2,
    )
    out = model(x, edge_index, batch=batch)

    assert out.shape == (num_nodes, out_channels)


def test_ordered_gnn_gate_monotonicity():
    """Ordered forget gate is monotonically non-increasing across blocks.

    This is the ordering constraint enforced by the cumulative
    product (Eq. (4) and Eq. (6) of arXiv:2302.01524): if neuron
    block j is released for updating, all outer blocks are released
    as well.
    """
    num_nodes, channels = 10, 16
    x = torch.randn(num_nodes, channels)
    m = torch.randn(num_nodes, channels)

    layer = OrderedGNNLayer(in_channels=channels, out_channels=channels)
    f_hat, i_hat = layer.ordered_gates(torch.cat([x, m], dim=-1))

    assert f_hat.shape == (num_nodes, channels)
    assert i_hat.shape == (num_nodes, channels)
    # Cumulative product of values in (0, 1) must never increase.
    assert (f_hat[:, 1:] <= f_hat[:, :-1] + 1e-7).all()
    # Gates must stay in the valid (0, 1) range.
    assert (f_hat >= 0).all() and (f_hat <= 1).all()
    assert (i_hat >= 0).all() and (i_hat <= 1).all()


def test_ordered_gnn_chunk_size():
    """Chunked gating shares one gate per block of chunk_size neurons."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = OrderedGNN(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=out_channels,
        num_layers=2,
        chunk_size=4,
    )
    out = model(x, edge_index)

    assert model.convs[0].num_gates == 4
    assert out.shape == (num_nodes, out_channels)

    # Gates expanded by the chunking trick are constant within chunks.
    fused = torch.randn(num_nodes, 2 * 16)
    f_hat, _ = model.convs[0].ordered_gates(fused)
    blocks = f_hat.view(num_nodes, 4, 4)
    assert torch.allclose(blocks, blocks[:, :, :1].expand_as(blocks))


def test_ordered_gnn_invalid_chunk_size():
    """chunk_size not dividing the hidden dimension raises ValueError."""
    with pytest.raises(ValueError, match="divisible"):
        OrderedGNNLayer(in_channels=16, out_channels=16, chunk_size=5)


def test_ordered_gnn_layer_projection():
    """A layer with in_channels != out_channels projects the skip path."""
    num_nodes, in_channels, out_channels = 10, 8, 12
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    layer = OrderedGNNLayer(
        in_channels=in_channels, out_channels=out_channels
    )
    out = layer(x, edge_index)

    assert isinstance(layer.lin_skip, torch.nn.Linear)
    assert out.shape == (num_nodes, out_channels)


def test_ordered_gnn_aggregation_schemes():
    """Alternative aggregation schemes run and keep the output shape."""
    num_nodes, in_channels, out_channels = 10, 8, 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    for aggr in ("mean", "add", "max"):
        model = OrderedGNN(
            in_channels=in_channels,
            hidden_channels=16,
            out_channels=out_channels,
            num_layers=2,
            aggr=aggr,
        )
        out = model(x, edge_index)
        assert out.shape == (num_nodes, out_channels)
