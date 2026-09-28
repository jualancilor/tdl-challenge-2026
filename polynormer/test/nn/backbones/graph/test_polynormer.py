"""Unit tests for the Polynormer backbone."""

import pytest
import torch

from topobench.nn.backbones.graph.polynormer import (
    GlobalAttention,
    Polynormer,
)


def _make_graph(num_nodes=5, num_edges=10, in_channels=8, seed=0):
    """Create a small synthetic graph for testing.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Number of (directed) edges in the graph.
    in_channels : int
        Dimension of the random node features.
    seed : int
        Seed controlling the random feature/edge generation.

    Returns
    -------
    tuple of torch.Tensor
        The node feature matrix and the edge index tensor.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, in_channels, generator=generator)
    edge_index = torch.randint(
        0, num_nodes, (2, num_edges), generator=generator
    )
    return x, edge_index


def test_forward_output_shape():
    """Forward pass on minimal data returns (num_nodes, out_channels)."""
    x, edge_index = _make_graph(num_nodes=5, num_edges=10, in_channels=8)
    model = Polynormer(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        num_layers=4,
        num_poly_layers=2,
        poly_degree=3,
        heads=2,
    )
    out = model(x, edge_index)
    assert out.shape == (5, 4)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_gradient_flow():
    """Backward pass runs and active-path parameters receive gradients."""
    x, edge_index = _make_graph()
    model = Polynormer(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        num_layers=4,
        num_poly_layers=2,
        use_global_attn=True,
    )
    out = model(x, edge_index)
    loss = out.sum()
    loss.backward()  # must not raise

    # With global attention enabled every parameter lies on the active path.
    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing == [], f"Parameters without gradient: {missing}"


def test_device_consistency():
    """Output device matches the input device."""
    x, edge_index = _make_graph()
    model = Polynormer(
        in_channels=8, hidden_channels=16, out_channels=4
    )
    out = model(x, edge_index)
    assert out.device == x.device


def test_eval_mode_is_deterministic():
    """Model in eval() mode produces identical output across calls."""
    x, edge_index = _make_graph()
    model = Polynormer(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        dropout=0.5,
        in_dropout=0.5,
    )
    model.eval()
    out1 = model(x, edge_index)
    out2 = model(x, edge_index)
    assert torch.allclose(out1, out2)


def test_dropout_is_active_in_training():
    """Dropout makes training-mode outputs stochastic."""
    x, edge_index = _make_graph()
    model = Polynormer(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        dropout=0.5,
        in_dropout=0.5,
    )
    model.train()
    out1 = model(x, edge_index)
    out2 = model(x, edge_index)
    assert not torch.allclose(out1, out2)


def test_global_attention_ablation_shape():
    """use_global_attn=False still yields the correct output shape."""
    x, edge_index = _make_graph()
    model = Polynormer(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        use_global_attn=False,
    )
    out = model(x, edge_index)
    assert out.shape == (5, 4)
    # The local path must still be trainable.
    out.sum().backward()
    assert model.pred.weight.grad is not None
    assert model.local_convs[0].lin.weight.grad is not None


@pytest.mark.parametrize("poly_degree", [1, 2, 3])
def test_different_poly_degrees(poly_degree):
    """Different polynomial degrees all run and keep the output shape.

    Parameters
    ----------
    poly_degree : int
        Polynomial degree K to test.
    """
    x, edge_index = _make_graph()
    model = Polynormer(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        poly_degree=poly_degree,
    )
    out = model(x, edge_index)
    assert out.shape == (5, 4)
    assert model.poly_degree == poly_degree


def test_batch_vector_is_accepted():
    """Passing a dummy batch vector does not change the output shape."""
    x, edge_index = _make_graph(num_nodes=6, num_edges=12)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    model = Polynormer(
        in_channels=8, hidden_channels=16, out_channels=4
    )
    out_with_batch = model(x, edge_index, batch=batch)
    assert out_with_batch.shape == (6, 4)


def test_extra_kwargs_ignored():
    """Extra kwargs such as edge_weight are accepted and ignored."""
    x, edge_index = _make_graph()
    model = Polynormer(
        in_channels=8, hidden_channels=16, out_channels=4
    )
    out = model(
        x,
        edge_index,
        batch=None,
        edge_weight=torch.ones(edge_index.size(1)),
        some_unused=123,
    )
    assert out.shape == (5, 4)


def test_multiple_heads():
    """Multi-head configuration produces the correct output shape."""
    x, edge_index = _make_graph()
    for heads in (1, 2, 4):
        model = Polynormer(
            in_channels=8,
            hidden_channels=8,
            out_channels=4,
            heads=heads,
        )
        out = model(x, edge_index)
        assert out.shape == (5, 4)


def test_pre_ln_and_constant_beta():
    """pre_ln and a non-negative constant beta both run correctly."""
    x, edge_index = _make_graph()
    model = Polynormer(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        pre_ln=True,
        beta=0.5,
    )
    out = model(x, edge_index)
    assert out.shape == (5, 4)
    # Constant beta keeps the parameter buffer at its initial value.
    assert torch.allclose(model.betas, torch.full_like(model.betas, 0.5))


def test_reset_parameters_runs():
    """reset_parameters re-initialises both beta strategies without error."""
    x, edge_index = _make_graph()
    for beta in (-1.0, 0.5):
        model = Polynormer(
            in_channels=8,
            hidden_channels=16,
            out_channels=4,
            pre_ln=True,
            beta=beta,
        )
        model.reset_parameters()
        out = model(x, edge_index)
        assert out.shape == (5, 4)


def test_invalid_arguments_raise():
    """Invalid layer/degree configurations raise ValueError."""
    with pytest.raises(ValueError):
        Polynormer(8, 16, 4, num_layers=2, num_poly_layers=0)
    with pytest.raises(ValueError):
        Polynormer(8, 16, 4, num_layers=2, num_poly_layers=3)
    with pytest.raises(ValueError):
        Polynormer(8, 16, 4, poly_degree=0)


def test_empty_graph():
    """A graph with no edges still produces the correct output shape."""
    x = torch.randn(3, 8)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    model = Polynormer(
        in_channels=8, hidden_channels=16, out_channels=4
    )
    model.eval()
    out = model(x, edge_index)
    assert out.shape == (3, 4)


def test_global_attention_module_separate_qk():
    """GlobalAttention with separate Q/K projections runs and is linear-width."""
    width_input = torch.randn(7, 2 * 4)  # heads=4, hidden=2 -> width 8
    module = GlobalAttention(
        hidden_channels=2,
        heads=4,
        num_layers=2,
        beta=-1.0,
        dropout=0.0,
        qk_shared=False,
    )
    module.reset_parameters()
    out = module(width_input)
    assert out.shape == (7, 8)
    out.sum().backward()
    assert module.q_lins[0].weight.grad is not None


def test_global_attention_module_constant_beta():
    """GlobalAttention with constant beta initialises and runs correctly."""
    width_input = torch.randn(5, 6)  # heads=3, hidden=2 -> width 6
    module = GlobalAttention(
        hidden_channels=2,
        heads=3,
        num_layers=1,
        beta=0.25,
        dropout=0.0,
    )
    out = module(width_input)
    assert out.shape == (5, 6)
    module.reset_parameters()
    assert torch.allclose(module.betas, torch.full_like(module.betas, 0.25))
