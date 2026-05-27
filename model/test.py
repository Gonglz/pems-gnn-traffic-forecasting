""""from dataset import RFGraphDataset
ds = RFGraphDataset()
print("T, N, F =", ds.T, ds.N, ds.F)
d0 = ds.get(0)
print("x:", d0.x.shape)        # -> [N, F]
print("nbr5:", d0.nbr5.shape)  # -> [N, K5]
print("nbr15:", d0.nbr15.shape)
print("nbr30:", d0.nbr30.shape)
print("y5, y15, y30:", d0.y5.shape, d0.y15.shape, d0.y30.shape)
"""

""""from dataset import RFGraphDataset
ds = RFGraphDataset()
d0 = ds.get(0)
print("x:", d0.x.shape)        # 应该是 [4883, 6]
print("nbr5:", d0.nbr5.shape)  # [4883, K5]
print("nbr15:",d0.nbr15.shape)
print("nbr30:",d0.nbr30.shape)
print("y5:", d0.y5.shape)      # [4883,1]
"""
from dataset_full import RFGraphDatasetFull
ds = RFGraphDatasetFull()
print("T,N,F =", ds.T, ds.N, ds.F)
data0 = ds.get(0)
print("x", data0.x.shape,
      "nbr5", data0.nbr5.shape,
      "y5", data0.y5.shape)
