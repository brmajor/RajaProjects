#include <RAJA/RAJA.hpp>
#include <iostream>
#include <algorithm>
#include "Timer.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

//type alias for scalar values and the area
//of matrix to compute
using Type = float;
using View3D = RAJA::View<Type, RAJA::Layout<3>>;

//crate template to initialize matrix values
template <typename Mat>
void
init(Mat& a, Mat& b, int n)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
	a(i, j, k) = b(i, j, k) = static_cast<Type>(i + j + (n - k)) * 10 / (n);
      }
    }
  }
}

//execute the loops. Take in initialized matrices, size, and timesteps
template <typename Policy, typename Mat>
void
exec(Mat& a, Mat& b, int n, int tsteps)
{
  //capture loop bodies (kernels) in lambdas. must be by value
  auto bodyB = [=] RAJA_HOST_DEVICE (int i, int j, int k) {
    b(i, j, k) = a(i, j, k)
      + Type(0.125) * (a(i + 1, j, k) - Type(2.0) * a(i, j, k) + a(i - 1, j, k) +
		       a(i, j + 1, k) - Type(2.0) * a(i, j, k) + a(i, j - 1, k) +
		       a(i, j, k + 1) - Type(2.0) * a(i, j, k) + a(i, j, k - 1));
  };
  
  auto bodyA = [=] RAJA_HOST_DEVICE (int i, int j, int k) {
    a(i, j, k) = b(i, j, k)
      + Type(0.125) * (b(i + 1, j, k) - Type(2.0) * b(i, j, k) + b(i - 1, j, k) +
		       b(i, j + 1, k) - Type(2.0) * b(i, j, k) + b(i, j - 1, k) +
		       b(i, j, k + 1) - Type(2.0) * b(i, j, k) + b(i, j, k - 1));
  };  

  //create the range of values we're using, then make a tuple of them (b/c 3d)
  auto seg = RAJA::make_range(1, n - 1);
  auto segs = RAJA::make_tuple(seg, seg, seg);

  //for tsteps apply the current policy over appropriate lambda using tuple of ranges
  for (int t = 0; t < tsteps; ++t) { 
    RAJA::kernel<Policy> (segs, bodyB);
    RAJA::kernel<Policy> (segs, bodyA);
  }
}

int
main() {

  int n, tsteps;
  std::cout << "N ==> ";
  std::cin >> n;
  std::cout << "Tsteps ==> ";
  std::cin >> tsteps;

  using namespace RAJA::statement;

  //sequential
  Type* cpuA = new Type[n * n * n];
  {
    //create matrices and initialize
    Type* cpuB = new Type[n * n * n];
    View3D bCPU(cpuB, n, n, n);
    View3D aCPU(cpuA, n, n, n);
    init(aCPU, bCPU, n);

    //create nested loops with lambda as body
    //loop_exec tells compiler to optimize as if normal loop
    using CPUPolicy = RAJA::KernelPolicy<
      For<0, RAJA::loop_exec,
	  For<1, RAJA::loop_exec,
	      For<2, RAJA::loop_exec,
		  Lambda<0>>>>>;

    //time execution
    Timer<> t;
    t.start();
    exec<CPUPolicy>(aCPU, bCPU, n, tsteps);
    t.stop();
    std::cerr << "Sequential: " << t.getElapsedMs() << '\n';

    //clean-up
    delete[] cpuB;
  }

  //CUDA on GPU
  {
    Type* gpuA;
    Type* gpuB;
    //tell CUDA to automatically transfer data between GPU and CPU as needed
    cudaMallocManaged (&gpuA, n * n * n * sizeof(Type));
    cudaMallocManaged (&gpuB, n * n * n * sizeof(Type));
    View3D bGPU(gpuB, n, n, n);
    View3D aGPU(gpuA, n, n, n);
    init(aGPU, bGPU, n);

    //use blocking to improve efficiency
    using CUDAPolicy = RAJA::KernelPolicy<
      CudaKernel<
	Tile<0, tile_fixed<32>, RAJA::cuda_block_z_loop,
	     Tile<1, tile_fixed<32>, RAJA::cuda_block_y_loop,
		  Tile<2, tile_fixed<32>, RAJA::cuda_block_x_loop,
		       For<0, RAJA::cuda_thread_z_loop,
			   For<1, RAJA::cuda_thread_y_loop,
			       For<2, RAJA::cuda_thread_x_loop,
				   Lambda<0>>>>>>>>>;
    
    Timer<> t;
    exec<CUDAPolicy>(aGPU, bGPU, n, tsteps);
    t.stop();
    std::cerr << "CUDA: " << t.getElapsedMs() << '\n';
    std::cerr << "GPU ok? " << std::boolalpha << std::equal(cpuA, cpuA + n * n * n, gpuA, gpuA + n * n * n) << '\n';
  
    cudaFree (gpuA);
    cudaFree (gpuB);
  }

  //OpenMP on CPU - default 12 threads b/c run on 6 core processor
  {
    Type* ompA = new Type[n * n * n];
    Type* ompB = new Type[n * n * n];
    View3D bOMP(ompB, n, n, n);
    View3D aOMP(ompA, n, n, n);
    init(aOMP, bOMP, n);

    //run outer loop in parallel, run middle loop sequentially, vectorize inner loop
    using OMPPolicy = RAJA::KernelPolicy<
      For<0, RAJA::omp_parallel_for_exec,
	  For<1, RAJA::loop_exec,
	      For<2, RAJA::simd_exec,
		  Lambda<0>>>>>;
    
    Timer<> t;
    exec<OMPPolicy>(aOMP, bOMP, n, tsteps);
    t.stop();
    std::cerr << "OpenMP: " << t.getElapsedMs() << '\n';
    std::cerr << "OMP ok? " << std::boolalpha << std::equal(cpuA, cpuA + n * n * n, ompA, ompA + n * n * n) << '\n';

    delete[] ompA;
    delete[] ompB;
  }

  // Cleanup
  delete[] cpuA;
  
  return 0;
}
