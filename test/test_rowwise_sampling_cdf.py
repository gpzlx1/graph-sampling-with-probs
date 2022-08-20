import os
import torch
from bench import bench
from load_graph import load_reddit, load_ogbn_products

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

g, _, _, _, _ = load_reddit()
g = g.formats(['csr'])
csr = g.adj(scipy_fmt='csr')
seeds = torch.arange(0, 200000).long().cuda()
indptr = torch.tensor(csr.indptr).long().cuda()
indices = torch.tensor(csr.indices).long().cuda()
probs = torch.rand(indices.numel()).abs().float().cuda()
num_picks = 25
replace = True

@bench(True)
def bench_func(seeds, indptr, indices, probs, num_picks, replace):
    torch.ops.gswp.RowWiseSamplingProb_CDF(seeds, indptr, indices, probs,
                                             num_picks, replace)


bench_func(seeds, indptr, indices, probs, num_picks, replace)