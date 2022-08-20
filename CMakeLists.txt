cmake_minimum_required(VERSION 3.15.0)
project(gswp LANGUAGES CUDA CXX)

set(CMAKE_PREFIX_PATH "/home/gpzlx1/graph_sampling_with_probs/venv/lib/python3.8/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Define our library target
add_library(gswp SHARED
    src/test.cu
    src/rowwise_sampling.cu
    src/rowwise_sampling_ares.cu
    src/rowwise_sampling_cdf.cu
    src/rowwise_sampling_alias.cu
    )

# Todo: more architectures
set_target_properties(gswp PROPERTIES CUDA_ARCHITECTURES "86")

# Enable C++14
target_compile_features(gswp PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(gswp ${TORCH_LIBRARIES} ${CUDA_cusparse_LIBRARY})