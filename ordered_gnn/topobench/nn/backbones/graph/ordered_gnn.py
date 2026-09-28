"""Ordered GNN backbone implementation.

Reference
---------
Song, Y., Zhou, C., Wang, X., & Lin, Z. (2023).
Ordered GNN: Ordering Message Passing to Deal with Heterophily and
Over-smoothing.
International Conference on Learning Representations (ICLR 2023).
arXiv:2302.01524.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing


class OrderedGNNLayer(MessagePassing):
    """Single layer of Ordered GNN with cumulative gating.

    Aligns the rooted-tree hierarchy of message passing hops with
    ordered blocks of neurons via a cumulative product gating mechanism,
    as described in Section 2 of arXiv:2302.01524. Messages from the
    k-th hop may only overwrite the outermost neuron blocks, so that
    ego information and near-hop information are never mixed with
    far-hop information inside the same neurons.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    out_channels : int
        Output feature dimension.
    dropout : float, optional
        Dropout probability applied to the fused ego/message features
        before the gating networks. Default is 0.0.
    aggr : str, optional
        Aggregation scheme for message passing. Default is 'mean'.
    chunk_size : int, optional
        Number of neurons controlled by a single gate (the chunking
        trick of Section 2.5 in arXiv:2302.01524). Must divide
        ``out_channels``. If None, one gate is predicted per neuron.
        Default is None.

    Notes
    -----
    The layer implements the combine stage of Eq. (8) in
    arXiv:2302.01524 with the gated update of Eq. (5),

    ``h_v_new = f_hat * h_v + i_hat * c_tilde``,

    where the gates are constrained to be monotone across neuron
    blocks so that the nested rooted-tree hierarchy of Eq. (3) aligns
    with ordered split points satisfying Eq. (4). The monotone
    (ordering) constraint is enforced with the cumulative-product
    master gates adapted from ON-LSTM: ``f_hat = cumprod(f)`` is
    non-increasing across blocks and plays the role of the paper's
    ``cumax`` gate in Eq. (6), while the input gate
    ``i_hat = i * (1 - shift(f_hat))`` couples consecutive blocks in
    the same spirit as the soft-OR operator of Eq. (7): once the
    forget gate has released block j, every block after j is released
    as well, so if neuron block j is updated then all outer blocks are
    also updated.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        aggr: str = "mean",
        chunk_size: int | None = None,
    ):
        super().__init__(aggr=aggr)

        if chunk_size is not None and out_channels % chunk_size != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"chunk_size ({chunk_size})."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.chunk_size = chunk_size

        # Number of gates: one per chunk of neurons (Section 2.5 of
        # arXiv:2302.01524) or one per neuron if chunking is disabled.
        self.num_gates = (
            out_channels if chunk_size is None else out_channels // chunk_size
        )

        # Gating networks acting on the fused vector [h_v, m_v]
        # (the linear fusion f_xi of Eq. (6) in arXiv:2302.01524).
        self.lin_f = nn.Linear(2 * in_channels, self.num_gates)
        self.lin_i = nn.Linear(2 * in_channels, self.num_gates)
        # Candidate state network (combine stage of Eq. (8)).
        self.lin_c = nn.Linear(2 * in_channels, out_channels)

        # Project the previous hidden state only when dimensions differ,
        # so that the gated combine of Eq. (5) is well defined.
        if in_channels == out_channels:
            self.lin_skip = nn.Identity()
        else:
            self.lin_skip = nn.Linear(in_channels, out_channels, bias=False)

    def ordered_gates(
        self, fused: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the ordered forget and input gates.

        Parameters
        ----------
        fused : torch.Tensor
            Fused ego/message features of shape
            ``(num_nodes, 2 * in_channels)``.

        Returns
        -------
        tuple of torch.Tensor
            Ordered forget gate ``f_hat`` and ordered input gate
            ``i_hat``, both of shape ``(num_nodes, out_channels)``.

        Notes
        -----
        The cumulative product enforces the ordering constraint of
        Eq. (4) in arXiv:2302.01524: ``f_hat`` is monotonically
        non-increasing across neuron blocks, the differentiable
        counterpart of the paper's ``cumax`` gate in Eq. (6). The
        shifted coupling of ``i_hat`` mirrors the soft-OR operator of
        Eq. (7): block j accepts new content in proportion to how much
        the cumulative forget gate has decayed up to block j - 1.
        """
        # Master forget and input gates in (0, 1), one entry per block.
        f = torch.sigmoid(self.lin_f(fused))
        i = torch.sigmoid(self.lin_i(fused))

        # Cumulative product enforces the ordering constraint: since
        # every factor lies in (0, 1), f_hat decreases monotonically
        # across blocks, so releasing block j releases all outer blocks.
        f_hat = torch.cumprod(f, dim=-1)

        # Shift f_hat one block to the right (block 0 receives 0), so
        # i_hat[:, j] = i[:, j] * (1 - f_hat[:, j - 1]) and the first
        # block reduces to i_hat[:, 0] = i[:, 0].
        f_shifted = F.pad(f_hat[:, :-1], (1, 0), value=0.0)
        i_hat = i * (1.0 - f_shifted)

        if self.chunk_size is not None:
            # Chunking trick (Section 2.5): each gate controls a
            # contiguous block of chunk_size neurons.
            f_hat = f_hat.repeat_interleave(self.chunk_size, dim=-1)
            i_hat = i_hat.repeat_interleave(self.chunk_size, dim=-1)

        return f_hat, i_hat

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Run one step of ordered message passing.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(num_nodes, in_channels)``.
        edge_index : torch.Tensor
            Edge indices of shape ``(2, num_edges)``.

        Returns
        -------
        torch.Tensor
            Updated node features of shape ``(num_nodes, out_channels)``.

        Notes
        -----
        Implements the full layer update of Eq. (8) in
        arXiv:2302.01524: mean aggregation of neighbor states,
        ordered gating, and the gated combine of Eq. (5).
        """
        # Aggregate stage of Eq. (8): permutation-invariant reduction
        # of neighbor hidden states into the message m_v.
        m = self.propagate(edge_index, x=x)

        # Fuse ego state and aggregated message as input to the gating
        # and candidate networks (the [h_v; m_v] concatenation of
        # Eq. (6)).
        fused = torch.cat([x, m], dim=-1)
        fused = F.dropout(fused, p=self.dropout, training=self.training)

        # Ordered gates (cumulative product; see ordered_gates).
        f_hat, i_hat = self.ordered_gates(fused)

        # Candidate hidden state built from ego and message content.
        c_tilde = torch.tanh(self.lin_c(fused))

        # Gated combine of Eq. (5): inner blocks (large f_hat) preserve
        # the previous state, outer blocks (large i_hat) admit the
        # newly aggregated context.
        return f_hat * self.lin_skip(x) + i_hat * c_tilde


class OrderedGNN(nn.Module):
    """Ordered GNN model stacking layers with cumulative gating.

    Stacks ``num_layers`` :class:`OrderedGNNLayer` modules between an
    input and an output projection, so that the rooted-tree hierarchy
    grown by successive message passing steps (Eq. (3) of
    arXiv:2302.01524) stays aligned with ordered blocks of neurons in
    every node representation. This ordering lets the model deal with
    heterophily and over-smoothing without target-specific designs
    (Section 1 of arXiv:2302.01524).

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    hidden_channels : int
        Hidden feature dimension. If ``chunk_size`` is given it must be
        divisible by ``chunk_size`` for block splitting.
    out_channels : int
        Output feature dimension.
    num_layers : int, optional
        Number of Ordered GNN layers. Default is 4.
    dropout : float, optional
        Dropout rate applied inside every layer. Default is 0.0.
    residual : bool, optional
        Whether to add residual connections between layers.
        Default is True.
    layer_norm : bool, optional
        Whether to apply layer normalization after every layer.
        Default is True.
    aggr : str, optional
        Aggregation scheme for message passing. Default is 'mean'.
    chunk_size : int, optional
        Number of neurons controlled by a single gate (Section 2.5 of
        arXiv:2302.01524). If None, one gate is predicted per neuron.
        Default is None.

    Notes
    -----
    The input projection follows Eq. (1) of arXiv:2302.01524. Each
    stacked layer applies the ordered update of Eq. (8); layer
    normalization matches the reference implementation, and residual
    connections are an optional stabilizer for deep stacks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        residual: bool = True,
        layer_norm: bool = True,
        aggr: str = "mean",
        chunk_size: int | None = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm

        # Input projection into the node embedding space (Eq. (1)).
        self.lin_in = nn.Linear(in_channels, hidden_channels)

        # Stack of ordered message passing layers (Eq. (8)).
        self.convs = nn.ModuleList(
            OrderedGNNLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dropout=dropout,
                aggr=aggr,
                chunk_size=chunk_size,
            )
            for _ in range(num_layers)
        )

        # Optional per-layer normalization (used after every layer in
        # the reference implementation of arXiv:2302.01524).
        self.norms = (
            nn.ModuleList(
                nn.LayerNorm(hidden_channels) for _ in range(num_layers)
            )
            if layer_norm
            else None
        )

        # Output projection to the requested dimensionality.
        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the full Ordered GNN forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(num_nodes, in_channels)``.
        edge_index : torch.Tensor
            Edge indices of shape ``(2, num_edges)``.
        batch : torch.Tensor, optional
            Batch vector assigning each node to a graph, of shape
            ``(num_nodes,)``. Message passing acts on the disjoint
            union of graphs, so it is accepted for API compatibility
            and requires no special handling. Default is None.
        **kwargs : dict
            Additional arguments (not used).

        Returns
        -------
        torch.Tensor
            Output node features of shape ``(num_nodes, out_channels)``.

        Notes
        -----
        Applies the input projection of Eq. (1) in arXiv:2302.01524,
        ``num_layers`` ordered updates of Eq. (8), and a final linear
        projection. Because every layer keeps hop-k information in
        dedicated outer neuron blocks (Eqs. (3)-(4)), stacking many
        layers does not wash out ego information (over-smoothing) and
        far-hop noise cannot corrupt near-hop signal (heterophily).
        """
        # Project raw features into the hidden embedding space.
        h = self.lin_in(x)

        for layer_idx, conv in enumerate(self.convs):
            h_prev = h
            # Ordered message passing update for this hop (Eq. (8)).
            h = conv(h, edge_index)
            if self.norms is not None:
                # Normalize the combined state, as in the reference
                # implementation.
                h = self.norms[layer_idx](h)
            if self.residual:
                # Residual connection between consecutive layers.
                h = h + h_prev

        # Project the final hidden state to the output dimension.
        return self.lin_out(h)
