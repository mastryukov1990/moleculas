from typing import List


class Features:
    def __init__(self, values: List):
        self.values = values

        self.v2indx = {}
        unique_values = set()

        for ind, v in enumerate(values):
            self.v2indx[v] = ind
            unique_values.add(v)

        self.num_unique = len(unique_values)
        self.unique_values = list(unique_values)
