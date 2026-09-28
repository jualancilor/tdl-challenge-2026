"""Unit tests for the PCN wrapper."""

import torch
from torch_geometric.data import Data

from topobench.nn.backbones.cell.path_complex_network import PCN
from topobench.nn.wrappers import PCNWrapper
from topobench.transforms.liftings import PathComplexLifting


class TestPCNWrapper:
    """Test the PCNWrapper class."""

    def test_PCNWrapper(self):
        """Test the forward pass of the PCN wrapper."""
        channels = 4
        edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
        edge_index = torch.tensor(
            [[u for u, v in edges] + [v for u, v in edges],
             [v for u, v in edges] + [u for u, v in edges]],
            dtype=torch.long,
        )
        data = Data(
            x=torch.randn(4, channels),
            edge_index=edge_index,
            num_nodes=4,
            y=torch.tensor([0]),
        )
        data = PathComplexLifting(max_path_length=2).forward(data)
        data.batch_0 = torch.zeros(data.x_0.shape[0], dtype=torch.long)

        wrapper = PCNWrapper(
            PCN(
                in_channels=channels,
                hidden_channels=channels,
                out_channels=channels,
            ),
            out_channels=channels,
            num_cell_dimensions=3,
        )
        out = wrapper(data)

        for key in ["labels", "batch_0", "x_0", "x_1", "x_2"]:
            assert key in out
        assert out["x_0"].shape == (data.x_0.shape[0], channels)
        assert out["x_2"].shape == (data.x_2.shape[0], channels)
