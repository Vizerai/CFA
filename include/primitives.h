#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "src/cuPrintf.cu"
#include "matrix_info.h"
#include "primitives_device.h"
#include "matrix_ops_device.h"

#include "scan.h"
#include "sparse_update.inl"
#include "sparse.h"
#include "load.h"
#include "spmv.h"

namespace device
{

/////////////////////////////////////////////////////////////////////////
/////////////////  Entry Wrapper Functions  /////////////////////////////
/////////////////////////////////////////////////////////////////////////
void InitCuPrint()
{
	cudaPrintfInit();
}

template <typename VALUE_TYPE>
void FILL(	cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
			const VALUE_TYPE value,
			cudaStream_t &stream)
{
	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	FILL<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			TPC(&a[0]),
			value,
			int(a.size()));
}

template <typename VALUE_TYPE>
void AND_OP(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
			const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &c,
			cudaStream_t &stream)
{
	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	AND_OP<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
			TPC(&a[0]),
			TPC(&b[0]),
			TPC(&c[0]),
			int(a.size()));
}

template <typename VALUE_TYPE>
void get_indices(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
					cudaStream_t &stream)
{
	const size_t NUM_BLOCKS = 1;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	get_indices<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
			TPC(&a[0]),
			TPC(&b[0]),
			int(a.size()));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void count(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
			const VALUE_TYPE val,
			INDEX_TYPE *h_res,
			INDEX_TYPE *d_res,
			cudaStream_t &stream)
{
	const size_t NUM_BLOCKS = 1;
	const size_t BLOCK_SIZE = 512;

	count<INDEX_TYPE, VALUE_TYPE, 512> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			TPC(&a[0]),
			val,
			d_res,
			int(a.size()));

	cudaMemcpyAsync(h_res, d_res, sizeof(INDEX_TYPE), cudaMemcpyDeviceToHost, stream);
}

template <typename VALUE_TYPE>
void gather_reduce(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &indices,
					const int index,
					cudaStream_t &stream)
{
	const size_t NUM_BLOCKS = 1;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

#if DEBUG
	assert(a.size() == b.size());
#endif

	gather_reduce<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
			TPC(&a[0]),
			TPC(&b[0]),
			TPC(&indices[index]),
			int(a.size()));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void column_select(	const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s,
					const INDEX_TYPE index,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
					cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == s.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	column_select<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
			infoA.num_rows,
			TPC(&A.row_offsets[0]),
			TPC(&A.column_indices[0]),
			TPC(&s[index]),
			TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void column_select_if(	const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &cond,
						const INDEX_TYPE index,
						cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
						cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == s.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	column_select_if<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
			infoA.num_rows,
			TPC(&A.row_offsets[0]),
			TPC(&A.column_indices[0]),
			TPC(&s[index]),
			TPC(&cond[index]),
			TPC(&y[0]));
}

template <typename VALUE_TYPE>
void AccumVec(	cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
				const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
				cudaStream_t &stream)
{
	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	AccumVec<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
			TPC(&a[0]),
			TPC(&b[0]),
			int(a.size()));
}

template <typename VALUE_TYPE>
void InnerProductStore(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
						cusp::array1d<VALUE_TYPE, cusp::device_memory> &c,
						const int index,
						cudaStream_t &stream)
{
	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	InnerProductStore<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
			TPC(&a[0]),
			TPC(&b[0]),
			int(a.size()),
			TPC(&c[index]));
}

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void OuterProduct(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 					cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
// 					cudaStream_t &stream)
// {
// 	mat_info<INDEX_TYPE> info;
// 	get_matrix_info(mat, info);

// #if(DEBUG)
// 	assert(src.num_rows == infoDst.num_rows);
// 	assert(src.num_cols == infoDst.num_cols);
// #endif

// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREADS;

// 	//OuterProduct<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>
// }

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
						cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
						cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> info;
	get_matrix_info(mat, info);

#if(DEBUG)
	assert(info.num_rows == a.size());
	assert(info.num_cols == b.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	OuterProductAdd_ELL_B<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			TPC(&a[0]),
			TPC(&b[0]),
			TPC(&index_count[0]),
			info.num_rows,
			info.num_cols,
			info.num_cols_per_row,
			info.pitch,
			TPC(&mat.column_indices.values[0]),
			TPC(&mat.values.values[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
						hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
						cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> info;
	get_matrix_info(mat, info);

#if(DEBUG)
	assert(info.num_rows == a.size());
	assert(info.num_cols == b.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	OuterProductAdd_HYB_B<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			TPC(&a[0]),
			TPC(&b[0]),
			TPC(&index_count[0]),
			info.num_rows,
			info.num_cols,
			info.num_cols_per_row,
			info.pitch,
			TPC(&mat.row_sizes[0]),
			TPC(&mat.matrix.ell.column_indices.values[0]),
			TPC(&mat.matrix.coo.row_indices[0]),
			TPC(&mat.matrix.coo.column_indices[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
						dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
						cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> info;
	get_matrix_info(mat, info);

#if(DEBUG)
	assert(info.num_rows == a.size());
	assert(info.num_cols == b.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	OuterProductAdd_DELL_B<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			TPC(&a[0]),
			TPC(&b[0]),
			TPC(&index_count[0]),
			TPC(&(*mat.row_offsets)[0]),
			TPC(&(*mat.column_indices)[0]),
			TPC(&mat.row_sizes[0]),
			TPC(&mat.coo.row_indices[0]),
			TPC(&mat.coo.column_indices[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void ell_add(	cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
				cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &B,
				cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &C,
				cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA, infoB, infoC;

	get_matrix_info(A, infoA);
	get_matrix_info(B, infoB);
	get_matrix_info(C, infoC);

	//fix this
}

} //namespace device

#endif
