from typing import List

import torch
from torch.utils.data import Dataset

from lib.preprocessing.models.molecul_graph_builder.graph_base import Graph


class MoleculesDataset(Dataset):
    def __init__(self, graph_list: List[Graph], labels: List[int]):
        self.graph_list = graph_list
        self.labels = labels

    def __getitem__(self, index: int):
        return self.graph_list[index], torch.Tensor(self.labels[index])

    def __len__(self):
        return len(self.graph_list)
