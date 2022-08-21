#include "./cuda_ops.cuh"

template <typename IdType, int TILE_SIZE, int BLOCK_WARPS, int WARP_SIZE>
__global__ void _CSRRowWiseSampleKernel(
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType *const in_rows,
    const IdType *const in_ptr,
    const IdType *const in_cols,
    const IdType *const out_ptr,
    const IdType *const ares_ptr,
    const IdType *const sort_ares_idxs,
    IdType *const out_rows,
    IdType *const out_cols)
{
    // we assign one warp per row
    assert(blockDim.x == WARP_SIZE);
    assert(blockDim.y == BLOCK_WARPS);

    int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    const int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

    while (out_row < last_row)
    {
        const int64_t row = in_rows[out_row];
        const int64_t in_row_start = in_ptr[row];
        const int64_t out_row_start = out_ptr[out_row];
        const int64_t deg = in_ptr[row + 1] - in_row_start;

        if (deg > num_picks)
        {
            const int64_t ares_row_start = ares_ptr[out_row];
            for (int64_t idx = threadIdx.x; idx < num_picks; idx += WARP_SIZE)
            {
                // get in and out index, the in_idx is one of top num_picks A-Res value
                // corresponding index in input CSR.
                const int64_t out_idx = out_row_start + idx;
                const int64_t ares_idx = ares_row_start + idx;
                const int64_t in_idx = sort_ares_idxs[ares_idx];
                // copy permutation over
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

template <typename IdType, typename FloatType, int TILE_SIZE, int BLOCK_WARPS, int WARP_SIZE>
__global__ void _CSRAResValueKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType *const in_rows,
    const IdType *const in_ptr,
    const FloatType *const prob,
    const IdType *const ares_ptr,
    IdType *const ares_idxs,
    FloatType *const ares)
{
    assert(blockDim.x == WARP_SIZE);
    assert(blockDim.y == BLOCK_WARPS);
    int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    const int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

    curandStatePhilox4_32_10_t rng;
    curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

    while (out_row < last_row)
    {
        const int64_t row = in_rows[out_row];
        const int64_t in_row_start = in_ptr[row];
        const int64_t deg = in_ptr[row + 1] - in_row_start;
        // A-Res value needs to be calculated only if deg is greater than num_picks
        // in weighted rowwise sampling without replacement
        if (deg > num_picks)
        {
            const int64_t ares_row_start = ares_ptr[out_row];

            for (int64_t idx = threadIdx.x; idx < deg; idx += WARP_SIZE)
            {
                const int64_t in_idx = in_row_start + idx;
                const int64_t ares_idx = ares_row_start + idx;
                FloatType item_prob = prob[in_row_start + idx];
                // compute A-Res value
                ares[ares_idx] = static_cast<FloatType>(__powf(curand_uniform(&rng), 1.0f / item_prob));
                ares_idxs[ares_idx] = static_cast<IdType>(in_idx);
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
    torch::Tensor sub_indptr, temp_indptr;
    std::tie(sub_indptr, temp_indptr) = _GetSubAndTempIndptr<int64_t>(seeds, indptr, num_picks, replace);
    thrust::device_ptr<int64_t> sub_prefix(static_cast<int64_t *>(sub_indptr.data_ptr<int64_t>()));
    thrust::device_ptr<int64_t> temp_prefix(static_cast<int64_t *>(temp_indptr.data_ptr<int64_t>()));
    int nnz = sub_prefix[num_rows];
    int temp_len = temp_prefix[num_rows];

    torch::Tensor coo_row = torch::empty(nnz, seeds.options());
    torch::Tensor coo_col = torch::empty(nnz, indices.options());
    torch::Tensor temp = torch::empty(temp_len, probs.options());
    torch::Tensor temp_idxs = torch::empty(temp_len, indices.options());

    const uint64_t random_seed = 7777;
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_WARPS = BLOCK_SIZE / WARP_SIZE;
    //constexpr int TILE_SIZE = BLOCK_WARPS * 16;
    constexpr int TILE_SIZE = 1;
    if (replace)
    {
        printf("Not Implemented.\n");
    }
    else
    {
        const dim3 block(WARP_SIZE, BLOCK_WARPS);
        const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

        _CSRAResValueKernel<int64_t, float, TILE_SIZE, BLOCK_WARPS, WARP_SIZE><<<grid, block>>>(
            random_seed,
            num_picks,
            num_rows,
            seeds.data_ptr<int64_t>(),
            indptr.data_ptr<int64_t>(),
            probs.data_ptr<float>(),
            temp_indptr.data_ptr<int64_t>(),
            temp_idxs.data_ptr<int64_t>(),
            temp.data_ptr<float>());

        torch::Tensor sort_temp = torch::empty_like(temp);
        torch::Tensor sort_temp_idxs = torch::empty_like(temp_idxs);

        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        cub::DoubleBuffer<float> sort_keys(temp.data_ptr<float>(), sort_temp.data_ptr<float>());
        cub::DoubleBuffer<int64_t> sort_values(temp_idxs.data_ptr<int64_t>(), sort_temp_idxs.data_ptr<int64_t>());
        cub::DeviceSegmentedSort::SortPairsDescending(
            d_temp_storage,
            temp_storage_bytes,
            sort_keys,
            sort_values,
            temp_len,
            num_rows,
            temp_indptr.data_ptr<int64_t>(),
            temp_indptr.data_ptr<int64_t>() + 1);

        c10::Allocator *cuda_allocator = c10::cuda::CUDACachingAllocator::get();
        c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
        d_temp_storage = _temp_data.get();

        cub::DeviceSegmentedSort::SortPairsDescending(
            d_temp_storage,
            temp_storage_bytes,
            sort_keys,
            sort_values,
            temp_len,
            num_rows,
            temp_indptr.data_ptr<int64_t>(),
            temp_indptr.data_ptr<int64_t>() + 1);

        _CSRRowWiseSampleKernel<int64_t, TILE_SIZE, BLOCK_WARPS, WARP_SIZE><<<grid, block>>>(
            num_picks,
            num_rows,
            seeds.data_ptr<int64_t>(),
            indptr.data_ptr<int64_t>(),
            indices.data_ptr<int64_t>(),
            sub_indptr.data_ptr<int64_t>(),
            temp_indptr.data_ptr<int64_t>(),
            sort_values.Current(),
            coo_row.data_ptr<int64_t>(),
            coo_col.data_ptr<int64_t>());
    }

    return {coo_row, coo_col};
}

static auto registry =
    torch::RegisterOperators(
        "gswp::RowWiseSamplingProb_ARes(Tensor seeds, Tensor indptr, Tensor indices, Tensor probs, int num_pick, bool replace) -> Tensor[]",
        &RowWiseSamplingProb_ARes);