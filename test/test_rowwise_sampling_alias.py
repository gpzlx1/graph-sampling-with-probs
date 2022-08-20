import os
import torch
so_path = os.path.join("/home/gpzlx1/graph_sampling_with_probs/build", 'libgswp.so')
torch.ops.load_library(so_path)

indptr = torch.tensor([0, 0, 5]).long().cuda()
indices = torch.arange(0, 5).long().cuda()
seeds = torch.tensor([1]).long().cuda()
probs = torch.tensor([2, 1, 1, 1, 5]).float().abs().cuda()

for i in torch.ops.gswp.RowWiseSamplingProb_Alias(seeds, indptr, indices, probs, 5, True):
    print(i)