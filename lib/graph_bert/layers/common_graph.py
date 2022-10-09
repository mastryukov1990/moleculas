import torch


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
    implicit_attn: the output of K Q
    explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {"e_out": edges.data[edge_feat]}

    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {
            field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))
        }

    return func
