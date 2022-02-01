from itertools import chain
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import DimeNet, NNConv, Set2Set, radius_graph


class Net(torch.nn.Module):
    def __init__(self, n_tasks, num_features=11, dim=64):
        super().__init__()
        self.n_tasks = n_tasks
        self.dim = dim
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr="mean")
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)

        self._init_task_heads()

    def _init_task_heads(self):
        for i in range(self.n_tasks):
            setattr(self, f"head_{i}", torch.nn.Linear(self.dim, 1))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, data, return_representation=False):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        features = F.relu(self.lin1(out))
        logits = torch.cat(
            [getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)], dim=1
        )
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(
            self.lin0.parameters(),
            self.conv.parameters(),
            self.gru.parameters(),
            self.set2set.parameters(),
            self.lin1.parameters(),
        )

    def task_specific_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return self.lin1.parameters()
