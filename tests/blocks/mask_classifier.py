import torch

from lib.graph_bert.nets.mask_classifier import MaskClassifier, MaskClassifierConfig


def test_mask_classifier():
    config = MaskClassifierConfig(in_dim=10, out_dim=20, hidden_dim=30)
    batch = 2

    x_batch = torch.ones([batch, config.in_dim])

    net = MaskClassifier(config)
    target = net.forward(x_batch)

    assert list(target.shape) == [batch, config.out_dim]
