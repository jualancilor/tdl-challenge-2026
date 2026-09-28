"""Wrapper for the PCN model."""

from topobench.nn.wrappers.base import AbstractWrapper


class PCNWrapper(AbstractWrapper):
    r"""Wrapper for the PCN model (arXiv:2308.06838).

    This wrapper defines the forward pass of the model. The PCN model
    returns the embeddings of the path cells of ranks 0, 1 and, when
    the lifted path complex has dimension 2, rank 2.
    """

    def forward(self, batch):
        r"""Forward pass for the PCN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_0, x_1, x_2 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            x_2=batch.get("x_2"),
            incidence_1=batch.incidence_1,
            incidence_2=batch.get("incidence_2"),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        if x_2 is not None:
            model_out["x_2"] = x_2
        return model_out
