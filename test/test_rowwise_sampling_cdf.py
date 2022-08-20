import os
import torch

so_path = os.path.join("/home/gpzlx1/graph_sampling_with_probs/build",
                       'libgswp.so')
torch.ops.load_library(so_path)

indptr = torch.tensor([0, 10, 11, 20, 22, 22]).long().cuda()
indices = torch.arange(0, 22).long().cuda()
seeds = torch.tensor([2, 3, 1, 4]).long().cuda()
probs = torch.rand(22).float().abs().cuda()

for i in torch.ops.gswp.RowWiseSamplingProb_CDF(seeds, indptr, indices, probs,
                                                5, True):
    print(i)