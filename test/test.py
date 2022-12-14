import os
import torch
so_path = os.path.join("/home/gpzlx1/graph_sampling_with_probs/build", 'libgswp.so')
torch.ops.load_library(so_path)

indptr = torch.tensor([0, 10 , 11, 20, 22, 22]).long().cuda()
seeds = torch.tensor([2, 3, 1, 4]).long().cuda()
num_pick = 5

print(torch.ops.gswp.GetSubIndptr(seeds, indptr, num_pick, False))
print(torch.ops.gswp.GetSubIndptr(seeds, indptr, num_pick, True))


print(torch.ops.gswp.GetSubAndTempIndptr(seeds, indptr, num_pick, False))
print(torch.ops.gswp.GetSubAndTempIndptr(seeds, indptr, num_pick, True))