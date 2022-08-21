#include "./cuda_ops.cuh"
#include "warpselect/WarpSelect.cuh"

template <typename IdType, typename FloatType, int TILE_SIZE, int BLOCK_WARPS, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void _CSRRowWiseSampleKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType *const in_rows,
    const IdType *const in_ptr,
    const IdType *const in_cols,
    const FloatType *const prob,
    const IdType *const out_ptr,
    IdType *const out_rows,
    IdType *const out_cols)
{
    // we assign one warp per row
    assert(blockDim.x == WARP_SIZE);
    assert(blockDim.y == BLOCK_WARPS);

    __shared__ IdType warpselect_out_index[WARP_SIZE * BLOCK_WARPS];

    // init warpselect
    WarpSelect<
        FloatType,
        IdType,
        true, // produce largest values
        Comparator<FloatType>,
        NumWarpQ,
        NumThreadQ,
        WARP_SIZE * BLOCK_WARPS>
        heap(_Limits<FloatType>::getMin(), -1, num_picks);

    int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    const int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

    curandStatePhilox4_32_10_t rng;
    curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

    int laneid = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.y;
    IdType *warpselect_out_index_per_warp = warpselect_out_index + warp_id * WARP_SIZE;

    while (out_row < last_row)
    {
        const int64_t row = in_rows[out_row];
        const int64_t in_row_start = in_ptr[row];
        const int64_t deg = in_ptr[row + 1] - in_row_start;
        const int64_t out_row_start = out_ptr[out_row];
        // A-Res value needs to be calculated only if deg is greater than num_picks
        // in weighted rowwise sampling without replacement
        if (deg > num_picks)
        {
            heap.reset();
            int limit = roundDown(deg, WARP_SIZE);
            IdType i = laneid;

            for (; i < limit; i += WARP_SIZE)
            {
                const int64_t in_idx = in_row_start + i;
                FloatType item_prob = prob[in_row_start + i];
                FloatType ares_prob = __powf(curand_uniform(&rng), 1.0f / item_prob);
                heap.add(ares_prob, i);
            }

            if (i < deg)
            {
                const int64_t in_idx = in_row_start + i;
                FloatType item_prob = prob[in_row_start + i];
                FloatType ares_prob = __powf(curand_uniform(&rng), 1.0f / item_prob);
                heap.addThreadQ(ares_prob, i);
                i += WARP_SIZE;
            }

            heap.reduce();
            heap.writeOutV(warpselect_out_index_per_warp, num_picks);

            for (int64_t idx = laneid; idx < num_picks; idx += WARP_SIZE)
            {
                const int64_t out_idx = out_row_start + idx;
                const int64_t in_idx = warpselect_out_index_per_warp[idx] + in_row_start;
                out_rows[out_idx] = static_cast<IdType>(row);
                out_cols[out_idx] = in_cols[in_idx];
            }
        }
        else
        {
            for (int64_t idx = threadIdx.x; idx < deg; idx += WARP_SIZE)
            {
                // get in and out index
                const int64_t out_idx = out_row_start + idx;
                const int64_t in_idx = in_row_start + idx;
                // copy permutation over
                out_rows[out_idx] = static_cast<IdType>(row);
                out_cols[out_idx] = in_cols[in_idx];
            }
        }

        out_row += BLOCK_WARPS;
    }
}

std::vector<torch::Tensor> RowWiseSamplingProb_ARes(
    torch::Tensor seeds,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor probs,
    int64_t num_picks,
    bool replace)
{
    int num_rows = seeds.numel();
    torch::Tensor sub_indptr = _GetSubIndptr<int64_t>(seeds, indptr, num_picks, replace);
    thrust::device_ptr<int64_t> sub_prefix(static_cast<int64_t *>(sub_indptr.data_ptr<int64_t>()));
    int nnz = sub_prefix[num_rows];

    torch::Tensor coo_row = torch::empty(nnz, seeds.options());
    torch::Tensor coo_col = torch::empty(nnz, indices.options());

    const uint64_t random_seed = 7777;
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_WARPS = BLOCK_SIZE / WARP_SIZE;
    // constexpr int TILE_SIZE = BLOCK_WARPS * 16;
    constexpr int TILE_SIZE = 1;
    if (replace)
    {
        printf("Not Implemented.\n");
    }
    else
    {
        const dim3 block(WARP_SIZE, BLOCK_WARPS);
        const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

        _CSRRowWiseSampleKernel<int64_t, float, TILE_SIZE, BLOCK_WARPS, WARP_SIZE, 32, 2><<<grid, block>>>(
            random_seed,
            num_picks,
            num_rows,
            seeds.data_ptr<int64_t>(),
            indptr.data_ptr<int64_t>(),
            indices.data_ptr<int64_t>(),
            probs.data_ptr<float>(),
            sub_indptr.data_ptr<int64_t>(),
            coo_row.data_ptr<int64_t>(),
            coo_col.data_ptr<int64_t>());
    }

    return {coo_row, coo_col};
}

static auto registry =
    torch::RegisterOperators(
        "gswp::RowWiseSamplingProb_ARes(Tensor seeds, Tensor indptr, Tensor indices, Tensor probs, int num_pick, bool replace) -> Tensor[]",
        &RowWiseSamplingProb_ARes);