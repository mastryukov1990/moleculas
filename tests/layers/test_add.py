import torch

from lib.graph_bert.layers.layers.add import SumAddLayer


def test_sum_add_layer():
    x = torch.ones([2, 3])

    k = SumAddLayer.forward(x, x)
    assert (k == 2 * x).min()
