from typing import Tuple

import dgl
import dgl.function as fn
import numpy as np
import torch

from lib.graph_bert.layers.attention_blocks.base import (
    V_H,
    K_H,
    Q_H,
    SCORE,
    PROJ_E,
    MultiHeadAttentionLayer,
)
from lib.graph_bert.layers.common_graph import (
    src_dot_dst,
    scaling,
    imp_exp_attn,
    out_edge_features,
    exp,
)
from lib.logger import Logger

logger = Logger(__name__)

EPS = 1e-6


class MultiHeadAttentionLayerEdge(MultiHeadAttentionLayer):
    def propagate_attention(self, g: dgl.DGLHeteroGraph):
        # Compute attention score
        g.apply_edges(src_dot_dst(K_H, Q_H, SCORE))

        # scaling
        g.apply_edges(scaling(SCORE, np.sqrt(self.out_dim)))

        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn(SCORE, PROJ_E))

        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features(SCORE))

        # softmax
        g.apply_edges(exp(SCORE))

        # Send weighted values to target nodes
        eids = g.edges()
        logger.info(f"graph = {g}")
        g.send_and_recv(eids, fn.src_mul_edge(V_H, SCORE, V_H), fn.sum(V_H, "wV"))
        g.send_and_recv(eids, fn.copy_edge(SCORE, SCORE), fn.sum(SCORE, "z"))

    def reduce_attention(self, g: dgl.DGLHeteroGraph) -> Tuple:
        return (
            g.ndata["wV"] / (g.ndata["z"] + torch.full_like(g.ndata["z"], EPS)),
            g.edata["e_out"],
        )
