#pragma once
#include <torch/script.h>

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <curand_kernel.h>

#include <cub/cub.cuh>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#define MIN(x, y) ((x < y) ? x : y)
#define MAX(x, y) ((x > y) ? x : y)

#define BLOCK_SIZE 128

template <typename IdType>
inline void cub_exclusiveSum(
    IdType *arrays,
    const IdType array_length)
{
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        arrays,
        arrays,
        array_length);

    c10::Allocator *cuda_allocator = c10::cuda::CUDACachingAllocator::get();
    c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
    d_temp_storage = _temp_data.get();

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        arrays,
        arrays,
        array_length);
}

template <typename IdType>
inline torch::Tensor _GetSubIndptr(
    torch::Tensor seeds,
    torch::Tensor indptr,
    int64_t num_pick,
    bool replace)
{
    int64_t num_items = seeds.numel();
    torch::Tensor sub_indptr = torch::empty((num_items + 1), indptr.options());
    thrust::device_ptr<IdType> item_prefix(static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));

    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        thrust::device, it(0), it(num_items), [
                in = seeds.data_ptr<IdType>(),
                in_indptr = indptr.data_ptr<IdType>(),
                out = thrust::raw_pointer_cast(item_prefix),
                replace, num_pick
            ] __device__(int i) mutable {
            IdType row = in[i];
            IdType begin = in_indptr[row];
            IdType end = in_indptr[row + 1];
            if (replace)
            {
                out[i] = (end - begin) == 0 ? 0 : num_pick;
            }
            else
            {
                out[i] = MIN(end - begin, num_pick);
            }
        });

    cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), num_items + 1);
    return sub_indptr;
}

template <typename IdType>
inline std::pair<torch::Tensor, torch::Tensor> _GetSubAndTempIndptr(
    torch::Tensor seeds,
    torch::Tensor indptr,
    int64_t num_pick,
    bool replace)
{
    int64_t num_items = seeds.numel();
    torch::Tensor sub_indptr = torch::empty((num_items + 1), indptr.options());
    torch::Tensor temp_indptr = torch::empty((num_items + 1), indptr.options());
    thrust::device_ptr<IdType> sub_prefix(static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
    thrust::device_ptr<IdType> temp_prefix(static_cast<IdType*>(temp_indptr.data_ptr<IdType>()));

    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        thrust::device, it(0), it(num_items), [
                in = seeds.data_ptr<IdType>(),
                in_indptr = indptr.data_ptr<IdType>(),
                sub_ptr = thrust::raw_pointer_cast(sub_prefix),
                tmp_ptr = thrust::raw_pointer_cast(temp_prefix),
                replace, num_pick, num_items
            ] __device__(int i) mutable {
            IdType row = in[i];
            IdType begin = in_indptr[row];
            IdType end = in_indptr[row + 1];
            IdType deg = end - begin;
            if (replace)
            {   
                sub_ptr[i] = deg == 0 ? 0 : num_pick;
                tmp_ptr[i] = deg;
            }
            else
            {
                sub_ptr[i] = MIN(deg, num_pick);
                tmp_ptr[i] = deg > num_pick ? deg : 0;
            }
            if(i == num_items - 1){
                sub_ptr[num_items] = 0;
                tmp_ptr[num_items] = 0;
            }
        });

    cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(sub_prefix), num_items + 1);
    cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(temp_prefix), num_items + 1);
    return {sub_indptr, temp_indptr};
}

inline __device__ int64_t AtomicMax(
    int64_t * const address,
    const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = long long int; // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type*>(address),
                   static_cast<Type>(val));
}