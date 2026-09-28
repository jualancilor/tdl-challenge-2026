"""Test the graph-to-path-complex lifting (arXiv:2308.06838)."""

import torch
from torch_geometric.data import Data

from topobench.transforms import TRANSFORMS
from topobench.transforms.liftings import PathComplexLifting


def _make_graph(num_nodes, edges, feat_dim=4, seed=0):
    """Build a small undirected test graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : list of tuple
        Undirected edges as (u, v) pairs.
    feat_dim : int, optional
        Node feature dimension.
    seed : int, optional
        Seed for the node features.

    Returns
    -------
    torch_geometric.data.Data
        The test graph.
    """
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor(
        [
            [u for u, v in edges] + [v for u, v in edges],
            [v for u, v in edges] + [u for u, v in edges],
        ],
        dtype=torch.long,
    )
    return Data(
        x=torch.randn(num_nodes, feat_dim, generator=generator),
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=torch.tensor([0]),
    )


class TestPathComplexLifting:
    """Test the PathComplexLifting class."""

    def setup_method(self):
        """Initialise the PathComplexLifting class."""
        self.lifting = PathComplexLifting(max_path_length=2)

    def test_lifting_registered(self):
        """Test that the lifting is registered in TRANSFORMS."""
        assert TRANSFORMS["PathComplexLifting"] is PathComplexLifting

    def test_lifting_triangle(self):
        """Test the lifting of a triangle graph.

        For the triangle A-B-C (Section 3.1 of arXiv:2308.06838):

        - 3 vertices (rank 0),
        - 3 edges, i.e. paths of length 1 (rank 1),
        - 3 simple paths of length 2 (rank 2): of the 6 vertex
          orderings, a path and its reverse are the same cell for an
          undirected graph, leaving A-B-C, B-A-C and A-C-B.

        Every 2-path of the triangle has all 3 edges on its boundary:
        removing an endpoint yields an edge, and removing the interior
        vertex also yields an edge because the triangle closes.
        """
        data = _make_graph(3, [(0, 1), (1, 2), (2, 0)])
        lifted = self.lifting.forward(data.clone())

        # Hand-verified cell counts: 3 nodes, 3 edges, 3 paths of
        # length 2.
        assert lifted.shape == [3, 3, 3]
        assert lifted.x_0.shape[0] == 3
        assert lifted.x_1.shape[0] == 3
        assert lifted.x_2.shape[0] == 3
        assert lifted.incidence_1.shape == (3, 3)
        assert lifted.incidence_2.shape == (3, 3)
        # Every 2-path of a triangle is bounded by all three edges.
        assert torch.equal(lifted.incidence_2.to_dense(), torch.ones(3, 3))
        # Vertices keep their original features.
        assert torch.equal(lifted.x_0, data.x)

    def test_lifting_path_graph(self):
        """Test the lifting of a path (line) graph.

        The path graph 0-1-2-3 has 4 vertices, 3 edges, 2 simple paths
        of length 2 (0-1-2 and 1-2-3) and 1 simple path of length 3
        (0-1-2-3).
        """
        data = _make_graph(4, [(0, 1), (1, 2), (2, 3)])
        lifted = self.lifting.forward(data.clone())
        assert lifted.shape == [4, 3, 2]

        # With max_path_length=3 the single 3-path is also lifted.
        lifting_3 = PathComplexLifting(max_path_length=3)
        lifted_3 = lifting_3.forward(data.clone())
        assert lifted_3.shape == [4, 3, 2, 1]

    def test_lifting_boundary_relations(self):
        """Test the boundary maps of a hand-computed graph.

        The graph is a triangle 0-1-2 with a tail 2-3. Its cells, in
        the deterministic (lexicographic) cell order of the lifting:

        - rank 1: (0,1), (0,2), (1,2), (2,3);
        - rank 2: (0,1,2), (0,2,1), (0,2,3), (1,0,2), (1,2,3).

        Boundaries follow Definition 2 of arXiv:2308.06838 restricted
        to allowed paths (Definition 3): the three 2-paths inside the
        triangle have 3 boundary edges each (the interior removal
        yields the closing edge), while (0,2,3) and (1,2,3) have only
        their two endpoint removals as boundaries, since (0,3) and
        (1,3) are not edges.
        """
        data = _make_graph(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
        lifted = self.lifting.forward(data.clone())

        assert lifted.shape == [4, 4, 5]
        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert torch.equal(lifted.incidence_1.to_dense(), expected_incidence_1)
        expected_incidence_2 = torch.tensor(
            [
                [1.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        assert torch.equal(lifted.incidence_2.to_dense(), expected_incidence_2)

    def test_lifting_max_length_respected(self):
        """Test that no path cell exceeds max_path_length.

        A line graph of 6 nodes contains simple paths of length up to
        5, but with max_path_length=2 the lifted complex must stop at
        rank 2.
        """
        data = _make_graph(6, [(i, i + 1) for i in range(5)])
        lifted = self.lifting.forward(data.clone())
        # Ranks 0, 1 and 2 only.
        assert len(lifted.shape) == 3
        assert not hasattr(lifted, "incidence_3")
        assert not hasattr(lifted, "x_3")
        assert self.lifting.complex_dim == 2

    def test_lifting_deterministic(self):
        """Test that lifting the same graph twice is identical."""
        data = _make_graph(5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
        lifted_a = self.lifting.forward(data.clone())
        lifted_b = PathComplexLifting(max_path_length=2).forward(data.clone())
        for key in ["incidence_1", "incidence_2"]:
            assert torch.equal(
                lifted_a[key].to_dense(), lifted_b[key].to_dense()
            )
        for key in ["x_0", "x_1", "x_2", "shape"]:
            assert torch.equal(
                torch.as_tensor(lifted_a[key]),
                torch.as_tensor(lifted_b[key]),
            )

    def test_lifting_rank0_matches_node_order(self):
        """Test that the rank-0 axis stays in ascending node order.

        ``toponetx`` orders rank-0 cells lexicographically, so for more
        than ten nodes its row 2 is node ``10``. Node-level targets and
        masks keep the original node order, so a permuted rank-0 axis
        silently misaligns features and labels. Uses a path graph on 12
        nodes, where the two orderings differ.
        """
        n = 12
        edges = [(i, i + 1) for i in range(n - 1)]
        data = _make_graph(n, edges)
        # Distinct per-node features so a permutation is detectable.
        data.x = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        lifted = self.lifting.forward(data.clone())

        assert torch.equal(lifted.x_0, data.x), (
            "node features must not be permuted by the lifting"
        )

        # Two rows of incidence_1 share a column exactly when the
        # corresponding nodes share an edge, which pins the row order to
        # the node order without relying on the column order.
        b1 = lifted.incidence_1.to_dense()
        shares_edge = (b1 @ b1.T) != 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                assert bool(shares_edge[i, j]) == (abs(i - j) == 1), (
                    f"incidence_1 rows {i},{j} disagree with the graph"
                )

    def test_lifting_boundary_only_storage(self):
        """Test that only the boundary structure is materialized.

        The PWL scheme (Section 4 of arXiv:2308.06838) consumes
        boundary and upper neighborhoods, both derived from the
        boundary incidence matrices; laplacians and (co)adjacencies
        are quadratically large around hubs and are not stored.
        """
        data = _make_graph(3, [(0, 1), (1, 2), (2, 0)])
        lifted = self.lifting.forward(data.clone())
        assert hasattr(lifted, "incidence_1")
        assert hasattr(lifted, "incidence_2")
        for key in [
            "adjacency_1",
            "coadjacency_2",
            "down_laplacian_2",
            "up_laplacian_1",
            "hodge_laplacian_2",
        ]:
            assert not hasattr(lifted, key)

    def test_lifting_explicit_neighborhoods(self):
        """Test the fallback to the generic connectivity builder.

        When a model config requests explicit neighborhoods, the
        lifting delegates to the generic full connectivity builder.
        """
        data = _make_graph(3, [(0, 1), (1, 2), (2, 0)])
        lifting = PathComplexLifting(
            max_path_length=2,
            neighborhoods=["down_incidence-1", "down_incidence-2"],
        )
        lifted = lifting.forward(data.clone())
        assert torch.equal(
            lifted["down_incidence-2"].to_dense(), torch.ones(3, 3)
        )

    def test_lifting_preserves_edge_attr(self):
        """Test that edge features are kept when requested."""
        data = _make_graph(3, [(0, 1), (1, 2), (2, 0)])
        # Assign undirected edge features: both directions of an edge
        # share the feature (min endpoint, max endpoint).
        data.edge_attr = torch.zeros(6, 2)
        for e, (u, v) in enumerate(
            zip(*data.edge_index.tolist(), strict=True)
        ):
            data.edge_attr[e] = torch.tensor(
                [float(min(u, v)), float(max(u, v))]
            )
        lifting = PathComplexLifting(
            max_path_length=2, preserve_edge_attr=True
        )
        lifted = lifting.forward(data.clone())
        expected_x_1 = torch.tensor([[0.0, 1.0], [0.0, 2.0], [1.0, 2.0]])
        assert torch.equal(lifted.x_1, expected_x_1)
