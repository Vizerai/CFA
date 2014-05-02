#ifndef SPARSE_H
#define SPARSE_H

#include <cusp/array1d.h>
#include <cusp/print.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/inner_product.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

//cusp and thrust
typedef cusp::array1d<char, cusp::device_memory> 	CuspVectorChar_d;
typedef cusp::array1d<char, cusp::host_memory> 		CuspVectorChar_h;
typedef cusp::array1d<short, cusp::device_memory> 	CuspVectorShort_d;
typedef cusp::array1d<short, cusp::host_memory> 	CuspVectorShort_h;
typedef cusp::array1d<int, cusp::device_memory> 	CuspVectorInt_d;
typedef cusp::array1d<int, cusp::host_memory> 		CuspVectorInt_h;
typedef cusp::array1d<double, cusp::device_memory> 	CuspVector_d;
typedef cusp::array1d<double, cusp::host_memory> 	CuspVector_h;
#define TPC(x)		thrust::raw_pointer_cast(x)

inline void safeSync()
{
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		fprintf( stderr, "!! GPU program execution error on line %d : cudaError=%d, (%s)\n", __LINE__, error, cudaGetErrorString(error));
}

// inline void checkCUDAError(const char *msg)
// {
// 	cudaError_t err = cudaGetLastError();
// 	if( cudaSuccess != err) 
// 	{
// 		fprintf(stderr, "Cuda error: %s: %s\n", msg, cudaGetErrorString(err));
// 		exit(EXIT_FAILURE);
// 	}
// }

// inline void checkCUDAError(const cudaError_t err, const char *msg)
// {
// 	if( cudaSuccess != err) 
// 	{
// 		fprintf(stderr, "Cuda error: %s: %s\n", msg, cudaGetErrorString(err));
// 		exit(EXIT_FAILURE);
// 	}
// }

#endif