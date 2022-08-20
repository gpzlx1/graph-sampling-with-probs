#include "./cuda_ops.cuh"

template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType *const in_rows,
    const IdType *const in_ptr,
    const IdType *const in_index,
    const IdType *const out_ptr,
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
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks)
    {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE)
      {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
      }
    }
    else
    {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE)
      {
        out_cols[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE)
      {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks)
        {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_cols + out_row_start + num, IdType(idx));
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE)
      {
        const IdType perm_idx = out_cols[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
      }
    }
    out_row += 1;
  }
}

template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType *const in_rows,
    const IdType *const in_ptr,
    const IdType *const in_index,
    const IdType *const out_ptr,
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
    const int64_t deg = in_ptr[row + 1] - in_row_start;

    if (deg > 0)
    {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE)
      {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start + edge];
      }
    }
    out_row += 1;
  }
}

std::vector<torch::Tensor> RowWiseSamplingUniform(
    torch::Tensor seeds,
    torch::Tensor indptr,
    torch::Tensor indices,
    int64_t num_picks,
    bool replace)
{
  int num_rows = seeds.numel();
  torch::Tensor sub_indptr = _GetSubIndptr<int64_t>(seeds, indptr, num_picks, replace);
  thrust::device_ptr<int64_t> item_prefix(static_cast<int64_t *>(sub_indptr.data_ptr<int64_t>()));
  int nnz = item_prefix[num_rows];

  torch::Tensor coo_row = torch::empty(nnz, seeds.options());
  torch::Tensor coo_col = torch::empty(nnz, indices.options());

  const uint64_t random_seed = 7777;

  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace)
  {
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    _CSRRowWiseSampleUniformReplaceKernel<int64_t, TILE_SIZE><<<grid, block>>>(
        random_seed,
        num_picks,
        num_rows,
        seeds.data_ptr<int64_t>(),
        indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),
        sub_indptr.data_ptr<int64_t>(),
        coo_row.data_ptr<int64_t>(),
        coo_col.data_ptr<int64_t>());
  }
  else
  {
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    _CSRRowWiseSampleUniformKernel<int64_t, TILE_SIZE><<<grid, block>>>(
        random_seed,
        num_picks,
        num_rows,
        seeds.data_ptr<int64_t>(),
        indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),
        sub_indptr.data_ptr<int64_t>(),
        coo_row.data_ptr<int64_t>(),
        coo_col.data_ptr<int64_t>());
  }

  return {coo_row, coo_col};
}

static auto registry =
    torch::RegisterOperators(
        "gswp::RowWiseSamplingUniform(Tensor seeds, Tensor indptr, Tensor indices, int num_pick, bool replace) -> Tensor[]",
        &RowWiseSamplingUniform);