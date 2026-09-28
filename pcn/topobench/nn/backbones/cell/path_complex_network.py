"""Path Complex Network (PCN) backbone implementation.

Reference
---------
Truong, Q., & Chin, P. (2024).
Weisfeiler and Lehman Go Paths: Learning Topological Features via Path
Complexes.
Proceedings of the AAAI Conference on Artificial Intelligence, 38(14),
15382-15391.
arXiv:2308.06838.

This implementation introduces the path-complex domain to TopoBench. The
message passing scheme follows the Path Weisfeiler-Lehman (PWL) test of
Section 4.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_mean_pool,
)


def upper_neighborhood_from_incidence(incidence):
    r"""Derive the upper neighborhood of a rank from an incidence matrix.

    Two rank-k cells :math:`\sigma, \tau` are upper adjacent when they
    are faces of a common rank-(k+1) cell :math:`\delta \in
    \mathcal{C}(\sigma, \tau)` (Definition 5 of arXiv:2308.06838). The
    PWL upper message of Section 4 of arXiv:2308.06838 is computed over
    the triples :math:`(\sigma, \tau, \delta)`, which this function
    enumerates from the sparse boundary incidence matrix
    :math:`B_{k+1}`.

    Parameters
    ----------
    incidence : torch.Tensor
        Sparse incidence matrix of shape (n_cells, n_cofaces) whose
        entry (i, j) is nonzero iff cell i is a face of coface j.

    Returns
    -------
    torch.Tensor
        Upper adjacency index of shape (2, n_pairs); messages flow from
        row 0 (:math:`\tau`) to row 1 (:math:`\sigma`).
    torch.Tensor
        Shared coface index :math:`\delta` of shape (n_pairs,), aligned
        with the columns of the upper adjacency index.
    """
    incidence = incidence.coalesce()
    device = incidence.device
    face, coface = incidence.indices()
    n_cofaces = incidence.size(1)
    # Group the faces of each coface contiguously.
    order = torch.argsort(coface, stable=True)
    face = face[order]
    counts = torch.bincount(coface, minlength=n_cofaces)
    group_start = counts.cumsum(0) - counts
    # A coface with c faces induces c * (c - 1) ordered pairs of
    # distinct faces (both directions of each upper adjacency).
    pairs_per_coface = counts * (counts - 1)
    n_pairs = int(pairs_per_coface.sum())
    if n_pairs == 0:
        return (
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    shared_coface = torch.repeat_interleave(
        torch.arange(n_cofaces, device=device), pairs_per_coface
    )
    pair_start = pairs_per_coface.cumsum(0) - pairs_per_coface
    local = torch.arange(n_pairs, device=device) - pair_start[shared_coface]
    c = counts[shared_coface]
    target_offset = local // (c - 1)
    source_offset = local % (c - 1)
    source_offset = source_offset + (source_offset >= target_offset).long()
    start = group_start[shared_coface]
    up_index = torch.stack(
        [face[start + source_offset], face[start + target_offset]]
    )
    return up_index, shared_coface


class PCNBoundaryConv(MessagePassing):
    r"""Boundary message passing of the PWL scheme.

    Computes the boundary message :math:`m_\mathcal{B}^{t+1}(\sigma) =
    \mathrm{AGG}_{\tau \in \mathcal{B}(\sigma)}\,
    M_\mathcal{B}(h_\sigma^t, h_\tau^t)` of Section 4 of
    arXiv:2308.06838, where :math:`\mathcal{B}(\sigma)` are the
    boundary cells of :math:`\sigma` (Definition 2 of arXiv:2308.06838,
    restricted to allowed paths). As in the reference implementation,
    :math:`M_\mathcal{B}` is the identity and the aggregation is a sum.

    Parameters
    ----------
    aggr : str, optional
        Aggregation scheme (default: "add").
    """

    def __init__(self, aggr="add"):
        super().__init__(aggr=aggr, flow="source_to_target", node_dim=0)

    def forward(self, x_face, x_cell, boundary_index):
        r"""Aggregate the boundary messages of every rank-k cell.

        Parameters
        ----------
        x_face : torch.Tensor
            Features of the rank-(k-1) cells, shape (n_faces, C).
        x_cell : torch.Tensor
            Features of the rank-k cells, shape (n_cells, C).
        boundary_index : torch.Tensor
            Boundary index of shape (2, n_incidences); row 0 holds face
            indices and row 1 the incident cell indices.

        Returns
        -------
        torch.Tensor
            Aggregated boundary messages, shape (n_cells, C).
        """
        return self.propagate(
            boundary_index,
            x=(x_face, x_cell),
            size=(x_face.size(0), x_cell.size(0)),
        )

    def message(self, x_j):
        r"""Boundary message from face :math:`\tau` to cell :math:`\sigma`.

        Parameters
        ----------
        x_j : torch.Tensor
            Features of the boundary cells :math:`\tau`.

        Returns
        -------
        torch.Tensor
            The boundary messages.
        """
        return x_j


class PCNUpperConv(MessagePassing):
    r"""Upper message passing of the PWL scheme.

    Computes the upper message :math:`m_\uparrow^{t+1}(\sigma) =
    \mathrm{AGG}_{\tau \in \mathcal{N}_\uparrow(\sigma),\, \delta \in
    \mathcal{C}(\sigma, \tau)}\, M_\uparrow(h_\sigma^t, h_\tau^t,
    h_\delta^t)` of Section 4 of arXiv:2308.06838. Following the
    reference implementation, :math:`M_\uparrow` concatenates the
    features of the upper-adjacent cell :math:`\tau` and of the shared
    coface :math:`\delta` and passes them through a single linear layer
    with a nonlinearity.

    By Theorem 11 of arXiv:2308.06838, co-boundary and lower-adjacent
    messages are redundant given boundary and upper-adjacent messages,
    hence PCN omits them.

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    aggr : str, optional
        Aggregation scheme (default: "add").
    """

    def __init__(self, channels, aggr="add"):
        super().__init__(aggr=aggr, flow="source_to_target", node_dim=0)
        self.msg_nn = nn.Linear(2 * channels, channels)

    def forward(self, x, x_coface, up_index, shared_coface):
        r"""Aggregate the upper messages of every rank-k cell.

        Parameters
        ----------
        x : torch.Tensor
            Features of the rank-k cells, shape (n_cells, C).
        x_coface : torch.Tensor
            Features of the rank-(k+1) cells, shape (n_cofaces, C).
        up_index : torch.Tensor
            Upper adjacency index of shape (2, n_pairs); messages flow
            from row 0 to row 1.
        shared_coface : torch.Tensor
            Index of the shared coface :math:`\delta` of every pair,
            shape (n_pairs,).

        Returns
        -------
        torch.Tensor
            Aggregated upper messages, shape (n_cells, C).
        """
        return self.propagate(
            up_index,
            x=x,
            coface_attr=x_coface[shared_coface],
            size=(x.size(0), x.size(0)),
        )

    def message(self, x_j, coface_attr):
        r"""Upper message from cell :math:`\tau` via coface :math:`\delta`.

        Parameters
        ----------
        x_j : torch.Tensor
            Features of the upper-adjacent cells :math:`\tau`.
        coface_attr : torch.Tensor
            Features of the shared cofaces :math:`\delta`.

        Returns
        -------
        torch.Tensor
            The upper messages
            :math:`M_\uparrow(h_\tau, h_\delta)`.
        """
        return F.relu(self.msg_nn(torch.cat([x_j, coface_attr], dim=-1)))


class PCNLayer(nn.Module):
    r"""One PWL message passing step over all ranks of a path complex.

    For every rank-k cell :math:`\sigma` the layer computes (Section 4
    of arXiv:2308.06838):

    .. math::
        m_\mathcal{B}^{t+1}(\sigma) &= \mathrm{AGG}_{\tau \in
            \mathcal{B}(\sigma)}\, M_\mathcal{B}(h_\sigma^t, h_\tau^t)\\
        m_\uparrow^{t+1}(\sigma) &= \mathrm{AGG}_{\tau \in
            \mathcal{N}_\uparrow(\sigma),\, \delta \in
            \mathcal{C}(\sigma, \tau)}\,
            M_\uparrow(h_\sigma^t, h_\tau^t, h_\delta^t)\\
        h_\sigma^{t+1} &= \mathrm{UP}(h_\sigma^t,
            m_\mathcal{B}^{t+1}(\sigma), m_\uparrow^{t+1}(\sigma))

    where, as in the reference implementation (a GIN-style injective
    update), :math:`\mathrm{UP}` perturbs the cell state by
    :math:`(1 + \epsilon)`, applies one MLP per message type and
    combines both branches with a final MLP.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    max_rank : int
        Maximum rank of the path complex.
    eps : float, optional
        GIN-style perturbation of the cell's own state (default: 0.0).
    """

    def __init__(self, in_channels, out_channels, max_rank, eps=0.0):
        super().__init__()
        self.max_rank = max_rank
        self.eps = eps
        self.boundary_convs = nn.ModuleList(
            PCNBoundaryConv() for _ in range(max_rank + 1)
        )
        # The maximum rank has no cofaces, hence no upper conv.
        self.upper_convs = nn.ModuleList(
            PCNUpperConv(in_channels) for _ in range(max_rank)
        )
        self.update_boundary_nns = nn.ModuleList(
            nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())
            for _ in range(max_rank + 1)
        )
        self.update_upper_nns = nn.ModuleList(
            nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())
            for _ in range(max_rank + 1)
        )
        self.combine_nns = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * out_channels, out_channels), nn.ReLU())
            for _ in range(max_rank + 1)
        )

    def forward(self, x_all, boundary_indices, up_neighborhoods):
        r"""Update the cells of every rank of the path complex.

        Parameters
        ----------
        x_all : list of torch.Tensor
            Per-rank cell features, ``x_all[k]`` of shape
            (n_cells_k, C).
        boundary_indices : list of torch.Tensor or None
            ``boundary_indices[k]`` is the (2, n_incidences) boundary
            index between ranks k-1 and k (None for k = 0).
        up_neighborhoods : list of tuple or None
            ``up_neighborhoods[k]`` is the (up_index, shared_coface)
            pair of rank k (None for the maximum rank).

        Returns
        -------
        list of torch.Tensor
            Updated per-rank cell features.
        """
        out = []
        for rank, x in enumerate(x_all):
            m_boundary = torch.zeros_like(x)
            if rank > 0 and boundary_indices[rank] is not None:
                m_boundary = self.boundary_convs[rank](
                    x_all[rank - 1], x, boundary_indices[rank]
                )
            m_upper = torch.zeros_like(x)
            if rank < len(x_all) - 1 and up_neighborhoods[rank] is not None:
                up_index, shared_coface = up_neighborhoods[rank]
                m_upper = self.upper_convs[rank](
                    x, x_all[rank + 1], up_index, shared_coface
                )
            out_boundary = self.update_boundary_nns[rank](
                (1 + self.eps) * x + m_boundary
            )
            out_upper = self.update_upper_nns[rank](
                (1 + self.eps) * x + m_upper
            )
            out.append(
                self.combine_nns[rank](
                    torch.cat([out_upper, out_boundary], dim=-1)
                )
            )
        return out


class PCN(nn.Module):
    r"""Path Complex Network (PCN) backbone.

    Stacks ``num_layers`` PWL message passing layers (Section 4 of
    arXiv:2308.06838) over a path complex of maximum rank
    ``max_path_length``, whose rank-k cells are the simple paths of
    length k of the underlying graph (Section 3.1 of arXiv:2308.06838).
    Messages are exchanged along the boundary and upper-adjacent
    neighborhoods only; by Theorem 11 of arXiv:2308.06838 the
    co-boundary and lower-adjacent neighborhoods carry no additional
    information.

    The forward pass follows TopoBench's cell-complex data convention:
    per-rank cell features together with the boundary incidence
    matrices produced by the ``PathComplexLifting`` transform, and
    returns the updated per-rank cell features. Complex-level pooling
    is exposed separately through :meth:`readout_complex`; inside
    TopoBench pipelines it is delegated to the model-level readout
    module.

    Parameters
    ----------
    in_channels : int
        Number of input channels of every rank.
    hidden_channels : int
        Number of hidden channels.
    out_channels : int
        Number of output channels of every rank.
    num_layers : int, optional
        Number of PWL message passing layers (default: 2).
    max_path_length : int, optional
        Maximum rank of the path cells (default: 2). TopoBench's
        cell-complex data convention carries ranks 0 to 2, hence values
        above 2 are not supported by this backbone.
    dropout : float, optional
        Dropout rate applied between layers (default: 0.0).
    readout : str, optional
        How per-rank representations are pooled by
        :meth:`readout_complex`, either "sum" or "mean"
        (default: "sum").

    Raises
    ------
    ValueError
        If ``max_path_length`` is not 1 or 2, or if ``readout`` is not
        "sum" or "mean".
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        max_path_length=2,
        dropout=0.0,
        readout="sum",
    ):
        super().__init__()
        if max_path_length not in (1, 2):
            raise ValueError(
                "TopoBench's cell-complex data convention carries ranks "
                f"0 to 2; got max_path_length={max_path_length}."
            )
        if readout not in ("sum", "mean"):
            raise ValueError(f"Invalid readout: {readout}.")
        self.max_path_length = max_path_length
        self.dropout = dropout
        self.readout = readout
        self.layers = nn.ModuleList(
            PCNLayer(
                in_channels if i == 0 else hidden_channels,
                out_channels if i == num_layers - 1 else hidden_channels,
                max_rank=max_path_length,
            )
            for i in range(num_layers)
        )

    def forward(self, x_0, x_1, x_2=None, incidence_1=None, incidence_2=None):
        r"""Forward pass of the PCN backbone.

        Parameters
        ----------
        x_0 : torch.Tensor
            Features of the rank-0 cells (vertices), shape
            (n_nodes, C).
        x_1 : torch.Tensor
            Features of the rank-1 cells (edges, i.e. paths of length
            1), shape (n_edges, C).
        x_2 : torch.Tensor, optional
            Features of the rank-2 cells (paths of length 2), shape
            (n_paths, C). Only used when ``max_path_length`` is 2.
        incidence_1 : torch.Tensor
            Sparse boundary incidence matrix between ranks 0 and 1,
            shape (n_nodes, n_edges).
        incidence_2 : torch.Tensor, optional
            Sparse boundary incidence matrix between ranks 1 and 2,
            shape (n_edges, n_paths). Only used when
            ``max_path_length`` is 2.

        Returns
        -------
        tuple of torch.Tensor
            The updated features ``(x_0, x_1, x_2)``; ``x_2`` is None
            when ``max_path_length`` is 1.
        """
        x_all = [x_0, x_1]
        incidences = [None, incidence_1]
        if self.max_path_length == 2:
            if x_2 is None:
                x_2 = x_0.new_zeros((0, x_0.size(1)))
            x_all.append(x_2)
            if incidence_2 is None:
                incidence_2 = torch.sparse_coo_tensor(
                    torch.empty(2, 0, dtype=torch.long, device=x_0.device),
                    torch.empty(0, device=x_0.device),
                    size=(x_1.size(0), 0),
                )
            incidences.append(incidence_2)

        boundary_indices = [None] + [
            incidence.coalesce().indices() for incidence in incidences[1:]
        ]
        up_neighborhoods = [
            upper_neighborhood_from_incidence(incidence)
            for incidence in incidences[1:]
        ] + [None]

        for i, layer in enumerate(self.layers):
            x_all = layer(x_all, boundary_indices, up_neighborhoods)
            if i < len(self.layers) - 1:
                x_all = [
                    F.dropout(x, p=self.dropout, training=self.training)
                    for x in x_all
                ]

        if self.max_path_length == 1:
            return x_all[0], x_all[1], None
        return x_all[0], x_all[1], x_all[2]

    def readout_complex(self, x_all, batch_indices=None):
        r"""Pool per-rank cell features into complex-level features.

        Implements the readout of the Path Isomorphism Network, whose
        architecture is identical to SIN and CIN (Section 4 of
        arXiv:2308.06838): the cells of every rank are pooled with the
        ``readout`` aggregation and the per-rank results are summed.

        Parameters
        ----------
        x_all : list of torch.Tensor
            Per-rank cell features; entries that are None or empty are
            skipped.
        batch_indices : list of torch.Tensor, optional
            Per-rank batch assignment vectors. If None, every rank is
            pooled into a single complex.

        Returns
        -------
        torch.Tensor
            Complex-level features of shape (batch_size, C).
        """
        pool = global_add_pool if self.readout == "sum" else global_mean_pool
        pooled = []
        for rank, x in enumerate(x_all):
            if x is None or x.size(0) == 0:
                continue
            batch = (
                batch_indices[rank]
                if batch_indices is not None
                else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            )
            pooled.append(pool(x, batch))
        return torch.stack(pooled, dim=0).sum(dim=0)
