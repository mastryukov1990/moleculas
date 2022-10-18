from dgl.dataloading import GraphDataLoader

from lib.graph_bert.nets.transform_block import GraphTransformBlockDefault
from lib.preprocessing.models.molecul_graph_builder.dgl_graph import FEATURE_COLUMN


def train_bert_epoch(net: GraphTransformBlockDefault, dataloader: GraphDataLoader):
    for g, label in dataloader:
        batch_h_index = g.ndata[FEATURE_COLUMN]
        batch_e_index = g.edata[FEATURE_COLUMN]

        h, e = net.forward(g, batch_h, batch_e)
