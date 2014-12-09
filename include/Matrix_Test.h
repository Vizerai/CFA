#ifndef MATRIX_TEST_H
#define MATRIX_TEST_H

#define NUM_GPUS                1
#define ARG_MAX                 512
#define BLOCKS                  26
#define BLOCK_THREADS           128
#define BLOCK_THREADS_MAX		1024
#define DEFAULT_OVERFLOW		256

#define MEMORY_ALIGNMENT    4096
#define ALIGN_UP(x,size)    ( ((size_t)x+(size-1))&(~(size-1)) ) //works for size that is a power of 2
#define ROUND_UP(x,y)       ( (x + y-1) / y )

#define CPU             0
#define GPU             1
#define BUILD_TYPE      GPU         //0 is CPU 1 is GPU

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

#if (BUILD_TYPE == GPU)
//CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#endif

//openmp
#include <omp.h>

//cusp
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/transpose.h>
#include <cusp/blas.h>
#include <cusp/print.h>

//dynamic ell
#include "macros.h"
#include "dell_matrix.h"

#if(BUILD_TYPE == GPU)

template <typename VALUE_TYPE>
void AND_OP(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
            const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
            cusp::array1d<VALUE_TYPE, cusp::device_memory> &c);

template <typename VALUE_TYPE>
void get_indices(   const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
                    cusp::array1d<VALUE_TYPE, cusp::device_memory> &b);

template <typename VALUE_TYPE>
void AccumVec(  cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
                const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void AccumMat(  const cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory> &mat,
                cusp::array1d<VALUE_TYPE, cusp::device_memory> &vec);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void column_select( const cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory> &A,
                    const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s,
                    const INDEX_TYPE index,
                    cusp::array1d<VALUE_TYPE, cusp::device_memory> &y);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProduct(  const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
                    const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
                    cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void ell_add(   cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
                cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &B,
                cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &C);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void ell_spmv(  const cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
                const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
                cusp::array1d<VALUE_TYPE, cusp::device_memory> &y);

#endif      //GPU

struct is_non_negative
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return (x >= 0);
    }
};

struct is_positive
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return (x > 0);
    }
};

#endif