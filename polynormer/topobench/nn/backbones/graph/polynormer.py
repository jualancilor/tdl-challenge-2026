"""Polynormer backbone implementation.

Polynormer is a polynomial-expressive graph transformer that runs in linear
time. It builds high-degree, permutation-equivariant polynomials on node
features by combining a *local* equivariant attention module (sparse,
GAT-style) with a *global* linear attention module (kernel-trick attention).
The polynomial expansion is realised through Hadamard products between a
feature projection and an attention-weighted aggregation, following Eq. (4),
Eq. (7) and Eq. (8) of the reference paper.

Reference
---------
Deng, C., Yue, Z., & Zhang, Z. (2024).
Polynormer: Polynomial-Expressive Graph Transformer in Linear Time.
International Conference on Learning Representations (ICLR 2024).
arXiv:2403.01232.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GlobalAttention(torch.nn.Module):
    """Linear-time equivariant global attention module.

    Implements the global attention layer of Eq. (8) in arXiv:2403.01232.
    Instead of materialising the dense ``n x n`` attention matrix, the module
    exploits the kernel trick of Eq. (6): the product ``softmax-like(Q) (K^T V)``
    is computed by first contracting keys with values, yielding ``O(n d^2)``
    complexity that is linear in the number of nodes ``n``.

    Parameters
    ----------
    hidden_channels : int
        Per-head hidden dimension.
    heads : int
        Number of attention heads. The working width is ``heads * hidden_channels``.
    num_layers : int
        Number of stacked global attention layers (``L_2`` in the paper).
    beta : float
        Initial value of the per-layer/per-channel mixing coefficient. If
        ``beta < 0`` the coefficients are learned from zero-initialised
        parameters and squashed with a sigmoid (matching the official code);
        otherwise they are kept constant at ``beta``.
    dropout : float
        Dropout probability applied after every global layer.
    qk_shared : bool, optional
        If True, queries and keys share the same projection (default), as in the
        official Polynormer release. Default is True.

    References
    ----------
    Eq. (6) and Eq. (8) of arXiv:2403.01232.
    """

    def __init__(
        self,
        hidden_channels: int,
        heads: int,
        num_layers: int,
        beta: float,
        dropout: float,
        qk_shared: bool = True,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared

        # Width of the concatenated multi-head representation.
        width = heads * hidden_channels

        # Per-layer mixing coefficient beta (controls H + 1 beta^T in Eq. (8)).
        if self.beta < 0:
            # Learned coefficients, later passed through a sigmoid.
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, width))
        else:
            # Constant coefficients.
            self.betas = torch.nn.Parameter(torch.ones(num_layers, width) * self.beta)

        self.h_lins = torch.nn.ModuleList()  # projection producing H (Eq. 8)
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()  # query projection W_Q
        self.k_lins = torch.nn.ModuleList()  # key projection W_K
        self.v_lins = torch.nn.ModuleList()  # value projection W_V
        self.lns = torch.nn.ModuleList()  # per-layer normalisation
        for _ in range(num_layers):
            self.h_lins.append(torch.nn.Linear(width, width))
            if not self.qk_shared:
                self.q_lins.append(torch.nn.Linear(width, width))
            self.k_lins.append(torch.nn.Linear(width, width))
            self.v_lins.append(torch.nn.Linear(width, width))
            self.lns.append(torch.nn.LayerNorm(width))
        self.lin_out = torch.nn.Linear(width, width)

    def reset_parameters(self):
        """Reset all trainable parameters of the module.

        Returns
        -------
        None
            None.
        """
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        # Re-initialise the mixing coefficients consistently with __init__.
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)
        self.lin_out.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stacked linear global attention layers.

        Computes Eq. (8) of arXiv:2403.01232 using the kernel trick of Eq. (6)
        so that the dense attention matrix is never formed.

        Parameters
        ----------
        x : torch.Tensor
            Node representations of shape ``(num_nodes, heads * hidden_channels)``.

        Returns
        -------
        torch.Tensor
            Updated node representations of shape
            ``(num_nodes, heads * hidden_channels)``.
        """
        seq_len = x.size(0)
        for i in range(self.num_layers):
            # H term of Eq. (8): a linear projection of the input.
            h = self.h_lins[i](x)
            # Keys are passed through sigmoid (sigma in Eq. 6) and split per head.
            k = torch.sigmoid(self.k_lins[i](x)).view(
                seq_len, self.hidden_channels, self.heads
            )
            # Queries share keys when qk_shared, otherwise own sigmoid projection.
            if self.qk_shared:
                q = k
            else:
                q = torch.sigmoid(self.q_lins[i](x)).view(
                    seq_len, self.hidden_channels, self.heads
                )
            # Values are linear (no nonlinearity), split per head.
            v = self.v_lins[i](x).view(seq_len, self.hidden_channels, self.heads)

            # Numerator of Eq. (6): first contract keys with values (K^T V),
            # then multiply by queries -> avoids the n x n attention matrix.
            kv = torch.einsum("ndh, nmh -> dmh", k, v)
            num = torch.einsum("ndh, dmh -> nmh", q, kv)

            # Denominator of Eq. (6): normalises attention scores per node so
            # that they sum to one (preserves the softmax normalisation).
            k_sum = torch.einsum("ndh -> dh", k)
            den = torch.einsum("ndh, dh -> nh", q, k_sum).unsqueeze(1)

            # Per-channel mixing coefficient (sigmoid if learned).
            if self.beta < 0:
                beta = torch.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)

            # Attention output, reshaped back to the concatenated head layout.
            x = (num / den).reshape(seq_len, -1)
            # Hadamard product with (H + beta) implements the polynomial term.
            x = self.lns[i](x) * (h + beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class Polynormer(torch.nn.Module):
    """Polynomial-expressive graph transformer backbone (Polynormer).

    The model first applies ``num_poly_layers`` local equivariant attention
    layers (Eq. (7)) that build a high-degree, permutation-equivariant
    polynomial on the node features using sparse GAT attention combined with a
    Hadamard product. The accumulated local representation can then be refined
    by ``num_layers - num_poly_layers`` global linear-attention layers (Eq. (8))
    when ``use_global_attn`` is enabled. Both stages run in linear time with
    respect to the number of nodes/edges.

    Polynomial degree. Each local layer raises the monomial degree of the
    learned polynomial through Hadamard products (Eq. (4)). The
    ``poly_degree`` parameter controls how many attention-weighted
    aggregations are multiplied together inside a single local layer; with
    ``poly_degree = 1`` the layer reduces to the canonical Polynormer local
    block ``H ⊙ (A V)`` of the official implementation.

    Parameters
    ----------
    in_channels : int
        Dimension of the input node features.
    hidden_channels : int
        Per-head hidden dimension. The internal working width is
        ``heads * hidden_channels``.
    out_channels : int
        Dimension of the output node representations.
    num_layers : int, optional
        Total number of layers (local + global). Default is 4.
    num_poly_layers : int, optional
        Number of local polynomial attention layers (``L_1``). The remaining
        ``num_layers - num_poly_layers`` layers form the global module
        (``L_2``). Default is 2.
    poly_degree : int, optional
        Polynomial degree ``K`` realised inside each local layer via iterated
        Hadamard products (Eq. (4)). Default is 3.
    dropout : float, optional
        Dropout probability used throughout the model. Default is 0.0.
    use_global_attn : bool, optional
        Whether to apply the global linear-attention module on top of the local
        representation. Disabling it recovers the local-only ablation. Default
        is True.
    heads : int, optional
        Number of attention heads. Default is 1.
    in_dropout : float, optional
        Dropout applied to the raw input features. Default is 0.15.
    beta : float, optional
        Initial value of the mixing coefficient ``beta``. A negative value
        (default) makes the coefficients learnable, matching the official
        Polynormer release. Default is -1.0.
    pre_ln : bool, optional
        Whether to apply layer normalisation before each local layer. Default
        is False.

    References
    ----------
    Eq. (4), Eq. (7) and Eq. (8) of arXiv:2403.01232.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_poly_layers: int = 2,
        poly_degree: int = 3,
        dropout: float = 0.0,
        use_global_attn: bool = True,
        heads: int = 1,
        in_dropout: float = 0.15,
        beta: float = -1.0,
        pre_ln: bool = False,
    ):
        super().__init__()

        if num_poly_layers < 1:
            raise ValueError("num_poly_layers must be >= 1.")
        if num_poly_layers > num_layers:
            raise ValueError("num_poly_layers cannot exceed num_layers.")
        if poly_degree < 1:
            raise ValueError("poly_degree must be >= 1.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_poly_layers = num_poly_layers
        # Global layers are whatever remains after the local polynomial layers.
        self.num_global_layers = num_layers - num_poly_layers
        self.poly_degree = poly_degree
        self.dropout = dropout
        self.in_drop = in_dropout
        self.use_global_attn = use_global_attn and self.num_global_layers > 0
        self.heads = heads
        self.beta = beta
        self.pre_ln = pre_ln

        # Concatenated multi-head width used internally.
        width = heads * hidden_channels

        # Per-local-layer mixing coefficient beta (Eq. 7, the 1 beta^T term).
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_poly_layers, width))
        else:
            self.betas = torch.nn.Parameter(
                torch.ones(num_poly_layers, width) * self.beta
            )

        # Local module components (one set per local layer).
        self.h_lins = torch.nn.ModuleList()  # projection producing H (Eq. 7)
        self.local_convs = torch.nn.ModuleList()  # sparse GAT attention A V
        self.lins = torch.nn.ModuleList()  # residual linear branch
        self.lns = torch.nn.ModuleList()  # per-layer normalisation
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        for _ in range(num_poly_layers):
            self.h_lins.append(torch.nn.Linear(width, width))
            # GATConv with concat=True outputs heads * hidden_channels features
            # and provides the sparse, neighbour-equivariant attention A.
            self.local_convs.append(
                GATConv(
                    width,
                    hidden_channels,
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    bias=False,
                )
            )
            self.lins.append(torch.nn.Linear(width, width))
            self.lns.append(torch.nn.LayerNorm(width))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(width))

        # Input projection in_channels -> heads * hidden_channels.
        self.lin_in = torch.nn.Linear(in_channels, width)
        # Normalisation applied to the local output before the global module.
        self.ln = torch.nn.LayerNorm(width)
        # Global linear-attention module (Eq. 8).
        self.global_attn = GlobalAttention(
            hidden_channels,
            heads,
            max(self.num_global_layers, 1),
            beta,
            dropout,
        )
        # Output head mapping the working width to out_channels. A single head
        # is used (rather than the official two-stage local/global heads) since
        # TopoBench performs a single forward pass into a downstream readout.
        self.pred = torch.nn.Linear(width, out_channels)

    def reset_parameters(self):
        """Reset all trainable parameters of the backbone.

        Returns
        -------
        None
            None.
        """
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.global_attn.reset_parameters()
        self.pred.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of Polynormer.

        Computes polynomial-expressive node representations via local
        equivariant attention (Eq. (7)) followed by optional global linear
        attention (Eq. (8)), as described in arXiv:2403.01232.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(num_nodes, in_channels)``.
        edge_index : torch.LongTensor
            Edge indices of shape ``(2, num_edges)``.
        batch : torch.LongTensor, optional
            Batch vector of shape ``(num_nodes,)`` assigning each node to a
            graph. Accepted for API compatibility with TopoBench; the linear
            attention is computed per node and is unaffected by it. Default is
            None.
        **kwargs : dict
            Additional keyword arguments (e.g. ``edge_weight``) accepted for
            compatibility with the TopoBench calling convention and ignored.

        Returns
        -------
        torch.Tensor
            Updated node representations of shape ``(num_nodes, out_channels)``.
        """
        # Input dropout on the raw features, then project to the working width.
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x = self.lin_in(x)  # in_channels -> heads * hidden_channels
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ----- Equivariant local attention (Eq. 7), accumulated over layers ---
        # X_local = sum_l X^(l): the running sum captures polynomials of
        # different degrees produced by successive local layers.
        x_local = torch.zeros_like(x)
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            # H term of Eq. (7): nonlinear feature projection.
            h = F.relu(self.h_lins[i](x))
            # A V term of Eq. (7): sparse GAT attention plus a linear residual.
            av = F.relu(local_conv(x, edge_index) + self.lins[i](x))
            av = F.dropout(av, p=self.dropout, training=self.training)
            # Polynomial expansion of degree ``poly_degree`` via iterated
            # Hadamard products (Eq. (4)); poly_degree=1 gives the canonical
            # Polynormer local block H ⊙ (A V).
            poly = h
            for _ in range(self.poly_degree):
                poly = poly * av
            # Per-channel mixing coefficient (1 beta^T term of Eq. (7)).
            if self.beta < 0:
                beta = torch.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            # Convex combination between the polynomial term and the residual.
            x = (1 - beta) * self.lns[i](poly) + beta * av
            x_local = x_local + x

        # ----- Equivariant global attention (Eq. 8) ---------------------------
        if self.use_global_attn:
            x_global = self.global_attn(self.ln(x_local))
            out = self.pred(x_global)
        else:
            out = self.pred(x_local)

        return out
