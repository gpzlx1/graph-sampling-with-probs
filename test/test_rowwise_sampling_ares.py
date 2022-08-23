import os
import torch
from bench import bench
from load_graph import load_reddit, load_ogbn_products, load_generate
import random

so_path = os.path.join("/home/gpzlx1/graph_sampling_with_probs/build",
                       'libgswp.so')
torch.ops.load_library(so_path)

indptr = torch.tensor([0, 5]).long().cuda()
indices = torch.arange(0, 5).long().cuda()
seeds = torch.tensor([0]).long().cuda()
probs = torch.rand(5).float().abs().cuda()

#for i in torch.ops.gswp.RowWiseSamplingProb_ARes(seeds, indptr, indices, probs,
#                                                 2, False):
#    print(i)



g, _, _, _, _ = load_generate(500000, 512)
#g, _, _, _, _ = load_reddit()
g = g.formats(['csr'])
csr = g.adj(scipy_fmt='csr')
seeds = [i for i in range(200000)]
random.shuffle(seeds)
seeds = torch.tensor(seeds).long().cuda()
indptr = torch.tensor(csr.indptr).long().cuda()
indices = torch.tensor(csr.indices).long().cuda()
probs = torch.rand(indices.numel()).abs().float().cuda()
num_picks = 25
replace = False

@bench(True)
def bench_func(seeds, indptr, indices, probs, num_picks, replace):
    torch.ops.gswp.RowWiseSamplingProb_ARes(seeds, indptr, indices, probs,
                                             num_picks, replace)


bench_func(seeds, indptr, indices, probs, num_picks, replace)
