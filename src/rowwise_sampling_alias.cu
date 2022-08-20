#include "./cuda_ops.cuh"

template <typename T>
struct pack_vector
{
    // Size and Ptr are shared by warps
    __device__ inline pack_vector(T *Ptr, int *Size_ptr, int Size, size_t Num_byte) : data(Ptr), size(Size_ptr)
    {
        *size = Size;
        capacity = int(Num_byte / sizeof(T));
        assert(*size <= capacity);
    }

    __device__ inline void Add(T t)
    {
        int old = atomicAdd(size, 1);
        assert(old < capacity);
        data[old] = t;
    }

    __device__ inline T Get(int id)
    {
        assert(id < capacity);
        return data[id];
    }

    __device__ inline bool Empty()
    {
        if (*size == 0)
            return true;
        return false;
    }

    __device__ inline T &operator[](int id)
    {
        assert(id < *size);
        return data[id];
    }

    __device__ inline int Size() { return *size; }

    int *size;
    int capacity;
    T *data;
};

template <typename FloatType, int group_size>
__device__ inline void prob_normalize(
    const FloatType *const prob,
    FloatType *_temp_prob,
    int64_t prob_len,
    cg::thread_block_tile<group_size> g)
{
    int laneid = g.thread_rank();
    FloatType thread_data = 0;
    FloatType sum = 0;
    for (int i = laneid; i < prob_len; i += g.size())
    {
        thread_data += prob[i];
    }

    sum = cg::reduce(g, thread_data, cg::plus<FloatType>());
    sum = g.shfl(sum, 0);

    for (int i = laneid; i < prob_len; i += g.size())
    {
        _temp_prob[i] = prob[i] * prob_len / sum;
    }
}

template <typename IdType, typename FloatType, int group_size>
__device__ inline void _consturctAliasTablePerGroups(
    pack_vector<IdType> large,
    pack_vector<IdType> small,
    pack_vector<IdType> alias,
    pack_vector<FloatType> probs,
    cg::thread_block_tile<group_size> g)
{
    int laneid = g.thread_rank();
    int g_size = g.size();
    for (int i = laneid; i < alias.Size(); i += g_size)
    {
        if (probs.Get(i) > 1)
        {
            large.Add(i);
        }
        else
        {
            small.Add(i);
        }
    }
    g.sync();

    while ((!small.Empty()) && (!large.Empty()))
    {
        int old_small_size = small.Size();
        int old_large_size = large.Size();
        uint tmp = min(old_small_size, old_large_size);
        uint act_size = min(group_size, tmp);
        bool act = laneid < act_size;
        g.sync();

        if (laneid == 0)
        {
            *small.size -= act_size;
            *large.size -= act_size;
        }
        g.sync();

        IdType smallV;
        IdType largeV;
        if (act)
        {
            smallV = small.Get(old_small_size + laneid - act_size);
            largeV = large.Get(old_large_size + laneid - act_size);
        }
        g.sync();

        // float old;
        if (act)
        {
            atomicAdd(&probs.data[largeV], probs.Get(smallV) - 1.0);
        }
        g.sync();

        if (act)
        {
            alias.data[smallV] = largeV;
            if (probs.Get(largeV) < 1.0)
            {
                small.Add(largeV);
            }
            else if (probs.Get(largeV) > 1.0)
            {
                large.Add(largeV);
            }
        }
        g.sync();
    }
}

template <typename IdType, typename FloatType, int TILE_SIZE, int group_num, int group_size>
__global__ void _CSRRowWiseSampleAliasReplaceKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType *const in_rows,
    const IdType *const in_ptr,
    const IdType *const in_cols,
    const FloatType *const prob,
    const IdType *const out_ptr,
    const IdType *const temp_ptr,
    IdType *const _temp_large,
    IdType *const _temp_small,
    IdType *const _temp_alias,
    FloatType *const _temp_probs,
    IdType *const out_rows,
    IdType *const out_cols)
{
    assert(group_size == blockDim.x);
    assert(group_num == blockDim.y);
    __shared__ int shared_size[4 * group_num];
    int *shared_size_per_group = shared_size + threadIdx.y * 4;
    __shared__ cg::experimental::block_tile_memory<4, 128> shared;
    cg::thread_block block = cg::experimental::this_thread_block(shared);
    cg::thread_block_tile<group_size> group = cg::experimental::tiled_partition<group_size>(block);
    int laneid = group.thread_rank();

    int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    const int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

    curandStatePhilox4_32_10_t rng;
    curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.y * group_size + threadIdx.x, 0, &rng);

    while (out_row < last_row)
    {
        const int64_t row = in_rows[out_row];
        const int64_t in_row_start = in_ptr[row];
        const int64_t deg = in_ptr[row + 1] - in_row_start;
        const int64_t out_row_start = out_ptr[out_row];
        const int64_t temp_row_start = temp_ptr[out_row];

        if (deg > 0)
        {
            // normalize
            prob_normalize<FloatType, group_size>(&prob[in_row_start], &_temp_probs[temp_row_start], deg, group);

            pack_vector<IdType> _large(_temp_large + temp_row_start, shared_size_per_group, 0, sizeof(IdType) * deg);
            pack_vector<IdType> _small(_temp_small + temp_row_start, shared_size_per_group + 1, 0, sizeof(IdType) * deg);
            pack_vector<IdType> _alias(_temp_alias + temp_row_start, shared_size_per_group + 2, deg, sizeof(IdType) * deg);
            pack_vector<FloatType> _probs(_temp_probs + temp_row_start, shared_size_per_group + 3, deg, sizeof(FloatType) * deg);

            // construct alias table
            _consturctAliasTablePerGroups<IdType, FloatType, group_size>(_large, _small, _alias, _probs, group);

            // select with alias table
            for (int64_t idx = laneid; idx < num_picks; idx += group.size())
            {
                int col = (int)floor(curand_uniform(&rng) * _alias.Size());
                float p = curand_uniform(&rng);
                int item = p < _probs[col] ? col : _alias[col];

                const int64_t in_idx = in_row_start + item;
                const int64_t out_idx = out_row_start + idx;
                out_rows[out_idx] = static_cast<IdType>(row);
                out_cols[out_idx] = in_cols[in_idx];
            }
        }
        out_row += group_num;
    }
}

std::vector<torch::Tensor> RowWiseSamplingProb_Alias(
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
    torch::Tensor temp_large = torch::empty(temp_size, seeds.options());
    torch::Tensor temp_small = torch::empty(temp_size, seeds.options());
    torch::Tensor temp_alias = torch::empty(temp_size, seeds.options());
    torch::Tensor temp_probs = torch::empty(temp_size, probs.options());

    const uint64_t random_seed = 7777;
    const int group_num = 1;
    const int group_size = 128 / group_num;
    constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
    if (replace)
    {
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
        _CSRRowWiseSampleAliasReplaceKernel<int64_t, float, TILE_SIZE, group_num, group_size><<<grid, block>>>(
            random_seed,
            num_picks,
            num_rows,
            seeds.data_ptr<int64_t>(),
            indptr.data_ptr<int64_t>(),
            indices.data_ptr<int64_t>(),
            probs.data_ptr<float>(),
            sub_indptr.data_ptr<int64_t>(),
            temp_indptr.data_ptr<int64_t>(),
            temp_large.data_ptr<int64_t>(),
            temp_small.data_ptr<int64_t>(),
            temp_alias.data_ptr<int64_t>(),
            temp_probs.data_ptr<float>(),
            coo_row.data_ptr<int64_t>(),
            coo_col.data_ptr<int64_t>());
    }
    else
    {
        printf("Not Implemented.\n");
    }

    return {coo_row, coo_col, temp_alias, temp_probs};
}

static auto registry =
    torch::RegisterOperators(
        "gswp::RowWiseSamplingProb_Alias(Tensor seeds, Tensor indptr, Tensor indices, Tensor probs, int num_pick, bool replace) -> Tensor[]",
        &RowWiseSamplingProb_Alias);