# model/dataset.py
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

DATA_DIR = '/scratch/lgong1/finalproject/pems_data'
NEIGHBOR_PKL = os.path.join(DATA_DIR, 'step62_neighbors.pkl')
PARQ_X      = os.path.join(DATA_DIR, 'X_ext.npy')
PARQ_Y      = os.path.join(DATA_DIR, 'Y.npy')

class RFGraphDataset(InMemoryDataset):
    def __init__(self):
        super().__init__(DATA_DIR)
        # 1) 加载特征和标签
        X = np.load(PARQ_X)  # (T, N, F)
        Y = np.load(PARQ_Y)  # (T, N)
        sids = np.load(os.path.join(DATA_DIR, 'sids.npy'))  # (4888,)
        payload = pickle.load(open(NEIGHBOR_PKL, 'rb'))
        graph_nodes = np.array(payload['graph_nodes'], dtype=int)  # (4883,)

        # —— 新增：构造 mask，只保留在 graph_nodes 里的站点
        mask = np.isin(sids, graph_nodes)  # Boolean (4888,) 有 neighbor 的为 True

        # —— 新增：过滤 X,Y 维度
        X = X[:, mask, :]  # -> (T, 4883, F)
        Y = Y[:, mask]  # -> (T, 4883)

        # 2) 赋值回去
        self.T, self.N, self.F = X.shape
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

        # 2) 读 in-memory precomputed neighbors
        payload = pickle.load(open(NEIGHBOR_PKL, 'rb'))
        # payload['graph_nodes'] 就是 station_id 列表，对应矩阵行
        nbrs = payload['neighbors']
        # 把每个列表做成固定宽度的 LongTensor，-1 作 padding
        self.nbr5  = self._to_padded_tensor(nbrs['5min'])
        self.nbr15 = self._to_padded_tensor(nbrs['15min'])
        self.nbr30 = self._to_padded_tensor(nbrs['30min'])

        # 3) 三个尺度的步长（单位：5min）
        self.delta5, self.delta15, self.delta30 = 1, 3, 6
        self.max_delta = max(self.delta5, self.delta15, self.delta30)

        # 4) 把所有时间步 t 构造成一个 Data list
        self.data_list = []
        for t in range(self.T - self.max_delta):
            data = Data(
                x = self.X[t],                                       # [N, F]
                nbr5  = self.nbr5,                                   # [N, K5]
                nbr15 = self.nbr15,                                  # [N, K15]
                nbr30 = self.nbr30,                                  # [N, K30]
                y5  = self.Y[t + self.delta5 ].unsqueeze(-1),        # [N,1]
                y15 = self.Y[t + self.delta15].unsqueeze(-1),
                y30 = self.Y[t + self.delta30].unsqueeze(-1),
            )
            self.data_list.append(data)

    def _to_padded_tensor(self, lists):
        """把 List[List[int]] -> LongTensor(N, K) 用 -1 padding"""
        N = len(lists)
        K = max(len(sub) for sub in lists)
        mat = torch.full((N, K), -1, dtype=torch.long)
        for i, row in enumerate(lists):
            mat[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        return mat

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
