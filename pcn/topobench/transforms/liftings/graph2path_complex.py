"""Graph-to-path-complex lifting transform.

Reference
---------
Truong, Q., & Chin, P. (2024).
Weisfeiler and Lehman Go Paths: Learning Topological Features via Path
Complexes.
Proceedings of the AAAI Conference on Artificial Intelligence, 38(14),
15382-15391.
arXiv:2308.06838.

This lifting introduces the path-complex domain to TopoBench. Path
complexes (Section 3 of arXiv:2308.06838) generalize both simplicial
complexes and regular cell complexes: rank-k cells are simple paths of
length k, i.e. ordered sequences of k+1 distinct pairwise-connected
vertices, rather than unordered vertex sets or glued disks.
"""

import numpy as np
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from toponetx.classes import PathComplex

from topobench.data.utils.utils import (
    generate_zero_sparse_connectivity,
    get_complex_connectivity,
)
from topobench.transforms.liftings.liftings import GraphLifting


class PathComplexLifting(GraphLifting):
    r"""Lift graphs to path complexes.

    Following Section 3.1 of arXiv:2308.06838, an undirected graph is
    lifted to the path complex whose rank-k cells are the simple paths
    of length k in the graph (Definition 1: an elementary p-path is a
    sequence of p+1 vertices; simple paths are the regular, allowed
    elementary paths whose vertices are distinct and consecutively
    connected by edges):

    - rank 0: the vertices,
    - rank 1: the edges (simple paths of length 1),
    - rank k: the simple paths of length k, for k up to
      ``max_path_length``.

    Since the graph is undirected, a path and its reverse are regarded
    as the same cell (Section 3.1 of arXiv:2308.06838); each cell is
    canonically oriented so that its first vertex is smaller than its
    last vertex.

    The boundary of a path cell follows the boundary operator of
    Definition 2 of arXiv:2308.06838 restricted to the allowed paths of
    the complex (Definition 3): removing any vertex of the sequence
    yields a face, provided the truncated sequence is itself a path of
    the graph. Removing an endpoint always yields a face; removing an
    interior vertex yields a face only when the two vertices adjacent
    to it are themselves connected by an edge (e.g. each 2-path of a
    triangle has three boundary edges, while a 2-path of a line graph
    has only two).

    Parameters
    ----------
    max_path_length : int, optional
        Maximum length of the lifted path cells, i.e. the maximum rank
        of the resulting path complex. Default is 2.
    signed : bool, optional
        Whether to return signed connectivity matrices. Default is
        False.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_path_length=2, signed=False, **kwargs):
        # The rank of a path cell equals its path length, hence the
        # dimension of the lifted complex equals ``max_path_length``.
        kwargs["complex_dim"] = max_path_length
        super().__init__(**kwargs)
        self.max_path_length = max_path_length
        self.signed = signed
        self.type = "graph2path_complex"

    def _get_path_complex_connectivity(self, path_complex) -> dict:
        r"""Build the connectivity structure of a lifted path complex.

        Only the boundary incidence matrices are materialized. The PWL
        message passing scheme (Section 4 of arXiv:2308.06838)
        exchanges messages along the boundary and upper-adjacent
        neighborhoods exclusively (co-boundary and lower-adjacent
        neighborhoods are redundant by Theorem 11 of arXiv:2308.06838),
        and both derive from the boundary matrices. The remaining Hodge
        structure (laplacians, (co)adjacencies) grows quadratically
        around high-degree hubs — a single hub of degree d induces
        O(d^2) upper-adjacent 2-path pairs — which makes it
        prohibitively large to precompute and store for heavy-tailed
        graphs.

        When ``neighborhoods`` are explicitly requested (by a model
        config), the generic full connectivity builder is used instead.

        Parameters
        ----------
        path_complex : toponetx.classes.PathComplex
            The lifted path complex.

        Returns
        -------
        dict
            The connectivity structure: per-rank boundary incidence
            matrices and the shape of the complex.
        """
        if self.neighborhoods is not None:
            return self._reindex_rank0(
                get_complex_connectivity(
                    path_complex,
                    self.complex_dim,
                    neighborhoods=self.neighborhoods,
                    signed=self.signed,
                ),
                path_complex,
            )
        practical_shape = list(
            np.pad(
                list(path_complex.shape),
                (0, self.complex_dim + 1 - len(path_complex.shape)),
            )
        )
        connectivity = {}
        for rank in range(self.complex_dim + 1):
            try:
                connectivity[f"incidence_{rank}"] = from_sparse(
                    path_complex.incidence_matrix(
                        rank=rank, signed=self.signed
                    )
                )
            except ValueError:
                connectivity[f"incidence_{rank}"] = (
                    generate_zero_sparse_connectivity(
                        m=practical_shape[rank - 1],
                        n=practical_shape[rank],
                    )
                )
        connectivity["shape"] = practical_shape
        return self._reindex_rank0(connectivity, path_complex)

    @staticmethod
    def _rank0_permutation(path_complex) -> torch.Tensor | None:
        """Return the permutation restoring ascending node order.

        ``toponetx`` indexes the rank-0 cells of a path complex in
        lexicographic order of their string representation, so with more
        than ten nodes row 2 of a rank-0 axis is node ``10`` rather than
        node ``2``. Node-level targets, masks and batch vectors all stay
        in the original node order, so the rank-0 axis has to be mapped
        back onto it; permuting the features instead would misalign them
        with those tensors.

        Parameters
        ----------
        path_complex : toponetx.classes.PathComplex
            The lifted path complex.

        Returns
        -------
        torch.Tensor or None
            Indices ``perm`` with ``perm[i]`` the current position of
            node ``i``, or ``None`` when the ordering is already
            ascending.
        """
        nodes = [path[0] for path in path_complex.skeleton(0)]
        if nodes == sorted(nodes):
            return None
        position = {node: ix for ix, node in enumerate(nodes)}
        return torch.tensor(
            [position[node] for node in sorted(nodes)], dtype=torch.long
        )

    @classmethod
    def _reindex_rank0(cls, connectivity: dict, path_complex) -> dict:
        """Map every rank-0 axis of ``connectivity`` onto node order.

        Rank-1 and higher axes are left untouched: they are internal to
        the complex and the features derived from them (``x_1``, ``x_2``)
        are produced from these same matrices, so they stay consistent.

        Parameters
        ----------
        connectivity : dict
            Connectivity matrices keyed as in the rest of TopoBench.
        path_complex : toponetx.classes.PathComplex
            The lifted path complex.

        Returns
        -------
        dict
            The connectivity, with rank-0 axes in ascending node order.
        """
        perm = cls._rank0_permutation(path_complex)
        if perm is None:
            return connectivity

        n_nodes = perm.numel()
        for key, value in connectivity.items():
            if key == "shape" or not torch.is_tensor(value):
                continue
            dense = value.to_dense() if value.is_sparse else value
            if dense.ndim != 2:
                continue
            # Rank-0 lives on the rows of ``incidence_1`` and on both
            # axes of any square rank-0 operator (adjacency, laplacian).
            rows_are_rank0 = key in {"incidence_0", "incidence_1"} or (
                key.endswith("_0") and dense.shape[0] == n_nodes
            )
            if rows_are_rank0 and dense.shape[0] == n_nodes:
                dense = dense[perm]
                if key.endswith("_0") and dense.shape[1] == n_nodes:
                    dense = dense[:, perm]
                connectivity[key] = dense.to_sparse()
        return connectivity

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift a graph to a path complex.

        Enumerates all simple paths of length at most
        ``max_path_length`` (Section 3.1 of arXiv:2308.06838) and
        builds the boundary structure of the resulting path complex,
        from which the Path Weisfeiler-Lehman message passing scheme
        (Section 4 of arXiv:2308.06838) draws its boundary and upper
        neighborhoods.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        path_complex = PathComplex(graph, max_rank=self.max_path_length)
        lifted_topology = self._get_path_complex_connectivity(path_complex)
        # The rank-0 axis has been mapped back onto ascending node order,
        # so node features carry over unpermuted and stay aligned with
        # the node-level targets and masks.
        lifted_topology["x_0"] = data.x
        # If new edges have been added during the lifting process, we
        # discard the edge attributes (mirrors Graph2CellLifting).
        if self.contains_edge_attr and len(path_complex.skeleton(1)) == (
            graph.number_of_edges()
        ):
            lifted_topology["x_1"] = torch.stack(
                [
                    graph.edges[edge[0], edge[1]]["features"]
                    for edge in path_complex.skeleton(1)
                ]
            )
        return lifted_topology


GRAPH2PATH_COMPLEX_LIFTINGS = {
    "PathComplexLifting": PathComplexLifting,
}
