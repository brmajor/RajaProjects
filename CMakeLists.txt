#cmake file author: Professor Killian

cmake_minimum_required(VERSION 3.10)
project(test LANGUAGES CXX CUDA)

set(CMAKE_C_COMPILER /opt/cuda/bin/gcc)
set(CMAKE_CXX_COMPILER /opt/cuda/bin/g++)
set(RAJA_DIR /opt/raja)
set(CMAKE_CXX_FLAGS "-O3 -march=native")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14;--expt-relaxed-constexpr;--expt-extended-lambda;--ptxas-options=-v;-arch=sm_61)

find_package(CUDA REQUIRED)
find_package(RAJA REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

## Remember: Add source files here
set_source_files_properties(heat-3d.cpp 
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

cuda_add_executable(heat-3d heat-3d.cpp)
target_compile_features(heat-3d PUBLIC cxx_std_17)
target_link_libraries(heat-3d RAJA)
target_include_directories(heat-3d PRIVATE ${RAJA_INCLUDE_DIR})

