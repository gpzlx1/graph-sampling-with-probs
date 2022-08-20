import os
import torch
from bench import bench
from load_graph import load_reddit, load_ogbn_products, load_generate
import random

so_path = os.path.join("/home/gpzlx1/graph_sampling_with_probs/build",
                       'libgswp.so')
torch.ops.load_library(so_path)

indptr = torch.tensor([0, 0, 5]).long().cuda()
indices = torch.arange(0, 5).long().cuda()
seeds = torch.tensor([1]).long().cuda()
probs = torch.tensor([2, 1, 1, 1, 5]).float().abs().cuda()

for i in torch.ops.gswp.RowWiseSamplingProb_Alias(seeds, indptr, indices,
                                                  probs, 5, True):
    print(i)

g, _, _, _, _ = load_generate(500000, 100)
g = g.formats(['csr'])
csr = g.adj(scipy_fmt='csr')
seeds = [i for i in range(200000)]
random.shuffle(seeds)
seeds = torch.tensor(seeds).long().cuda()
indptr = torch.tensor(csr.indptr).long().cuda()
indices = torch.tensor(csr.indices).long().cuda()
probs = torch.rand(indices.numel()).abs().float().cuda()
num_picks = 5
replace = True


@bench(True)
def bench_func(seeds, indptr, indices, probs, num_picks, replace):
    torch.ops.gswp.RowWiseSamplingProb_Alias(seeds, indptr, indices, probs,
                                             num_picks, replace)


bench_func(seeds, indptr, indices, probs, num_picks, replace)