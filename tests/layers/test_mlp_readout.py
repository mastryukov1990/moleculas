import torch

from lib.graph_bert.layers.mlp_readout_layer import MLPReadout, MLPReadoutConfig


def test_mlp_readout():
    batch = 2
    dim = 10
    out_dim = 100

    x_batch = torch.ones([batch, dim])

    config = MLPReadoutConfig(in_dim=dim, out_dim=out_dim)
    mlp = MLPReadout(config)

    target = mlp.forward(x_batch)
    assert list(target.shape) == [batch, config.out_dim]
