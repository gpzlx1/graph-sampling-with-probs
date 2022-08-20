#include "./cuda_ops.cuh"

template <typename FloatType>
struct BlockPrefixCallbackOp
{
    // Running prefix
    FloatType running_total;
    // Constructor
    __device__ BlockPrefixCallbackOp(FloatType running_total) : running_total(running_total) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ FloatType operator()(FloatType block_aggregate)
    {
        FloatType old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

template <typename IdType, typename FloatType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleReplaceKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType *const in_rows,
    const IdType *const in_ptr,
    const IdType *const in_cols,
    const FloatType *const prob,
    const IdType *const out_ptr,
    const IdType *const cdf_ptr,
    FloatType *const cdf,
    IdType *const out_rows,
    IdType *const out_cols)
{
    // we assign one warp per row
    assert(blockDim.x == BLOCK_SIZE);

    int64_t out_row = blockIdx.x * TILE_SIZE;
    const int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

    curandStatePhilox4_32_10_t rng;
    curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

    while (out_row < last_row)
    {
        const int64_t row = in_rows[out_row];
        const int64_t in_row_start = in_ptr[row];
        const int64_t out_row_start = out_ptr[out_row];
        const int64_t cdf_row_start = cdf_ptr[out_row];
        const int64_t deg = in_ptr[row + 1] - in_row_start;
        const FloatType MIN_THREAD_DATA = static_cast<FloatType>(0.0f);

        if (deg > 0)
        {
            // Specialize BlockScan for a 1D block of BLOCK_SIZE threads
            typedef cub::BlockScan<FloatType, BLOCK_SIZE> BlockScan;
            // Allocate shared memory for BlockScan
            __shared__ typename BlockScan::TempStorage temp_storage;
            // Initialize running total
            BlockPrefixCallbackOp<FloatType> prefix_op(MIN_THREAD_DATA);

            int64_t max_iter = (1 + (deg - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            // Have the block iterate over segments of items
            for (int64_t idx = threadIdx.x; idx < max_iter; idx += BLOCK_SIZE)
            {
                // Load a segment of consecutive items that are blocked across threads
                FloatType thread_data;
                if (idx < deg)
                    thread_data = prob[in_row_start + idx];
                else
                    thread_data = MIN_THREAD_DATA;
                thread_data = max(thread_data, MIN_THREAD_DATA);
                // Collectively compute the block-wide inclusive prefix sum
                BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, prefix_op);
                __syncthreads();

                // Store scanned items to cdf array
                if (idx < deg)
                {
                    cdf[cdf_row_start + idx] = thread_data;
                }
            }
            __syncthreads();

            for (int64_t idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE)
            {
                // get random value
                FloatType sum = cdf[cdf_row_start + deg - 1];
                FloatType rand = static_cast<FloatType>(curand_uniform(&rng) * sum);
                // get the offset of the first value within cdf array which is greater than random value.
                int64_t item = cub::UpperBound<FloatType *, int64_t, FloatType>(
                    &cdf[cdf_row_start], deg, rand);
                item = min(item, deg - 1);
                // get in and out index
                const int64_t in_idx = in_row_start + item;
                const int64_t out_idx = out_row_start + idx;
                // copy permutation over
                out_rows[out_idx] = static_cast<IdType>(row);
                out_cols[out_idx] = in_cols[in_idx];
            }
        }
        out_row += 1;
    }
}

std::vector<torch::Tensor> RowWiseSamplingProb_CDF(
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
    int temp_size = temp_prefix[num_rows];

    torch::Tensor coo_row = torch::empty(nnz, seeds.options());
    torch::Tensor coo_col = torch::empty(nnz, indices.options());
    torch::Tensor temp = torch::empty(temp_size, probs.options());

    const uint64_t random_seed = 7777;
    constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
    if (replace)
    {
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
        _CSRRowWiseSampleReplaceKernel<int64_t, float, TILE_SIZE><<<grid, block>>>(
            random_seed,
            num_picks,
            num_rows,
            seeds.data_ptr<int64_t>(),
            indptr.data_ptr<int64_t>(),
            indices.data_ptr<int64_t>(),
            probs.data_ptr<float>(),
            sub_indptr.data_ptr<int64_t>(),
            temp_indptr.data_ptr<int64_t>(),
            temp.data_ptr<float>(),
            coo_row.data_ptr<int64_t>(),
            coo_col.data_ptr<int64_t>());
    }
    else
    {
        printf("Not Implemented.\n");
    }

    return {coo_row, coo_col};
}

static auto registry =
    torch::RegisterOperators(
        "gswp::RowWiseSamplingProb_CDF(Tensor seeds, Tensor indptr, Tensor indices, Tensor probs, int num_pick, bool replace) -> Tensor[]",
        &RowWiseSamplingProb_CDF);