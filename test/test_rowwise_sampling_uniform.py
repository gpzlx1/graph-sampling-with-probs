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
num_pick = 5

for i in torch.ops.gswp.RowWiseSamplingUniform(seeds, indptr, indices,
                                               num_pick, False):
    print(i)

for i in torch.ops.gswp.RowWiseSamplingUniform(seeds, indptr, indices,
                                               num_pick, True):
    print(i)

g, _, _, _, _ = load_reddit()
g = g.formats(['csr'])
csr = g.adj(scipy_fmt='csr')
seeds = torch.arange(0, 200000).long().cuda()
indptr = torch.tensor(csr.indptr).long().cuda()
indices = torch.tensor(csr.indices).long().cuda()
num_picks = 25
replace = True


@bench(True)
def bench_func(seeds, indptr, indices, num_picks, replace):
    torch.ops.gswp.RowWiseSamplingUniform(seeds, indptr, indices, num_picks,
                                           replace)


bench_func(seeds, indptr, indices, num_picks, replace)