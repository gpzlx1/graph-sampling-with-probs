#include "./cuda_ops.cuh"

torch::Tensor GetSubIndptr(
    torch::Tensor seeds,
    torch::Tensor indptr,
    int64_t num_pick,
    bool replace)
{
    return _GetSubIndptr<int64_t>(seeds, indptr, num_pick, replace);
}


std::vector<torch::Tensor> GetSubAndTempIndptr(
    torch::Tensor seeds,
    torch::Tensor indptr,
    int64_t num_pick,
    bool replace)
{
    torch::Tensor sub_indptr, temp_indptr;
    std::tie(sub_indptr, temp_indptr) = _GetSubAndTempIndptr<int64_t>(seeds, indptr, num_pick, replace);
    return {sub_indptr, temp_indptr};
}

static auto registry =
    torch::RegisterOperators(
        "gswp::GetSubIndptr(Tensor seeds, Tensor indptr, int num_pick, bool replace) -> Tensor sub_indptr", &GetSubIndptr)
        .op("gswp::GetSubAndTempIndptr(Tensor seeds, Tensor indptr, int num_pick, bool replace) -> Tensor[]", &GetSubAndTempIndptr);