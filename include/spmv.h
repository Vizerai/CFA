#ifndef SPMV_H
#define SPMV_H

#include "spmv.inl"

namespace device
{

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_ellb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
            infoA.num_cols_per_row,
            infoA.pitch,
            TPC(&A.column_indices.values[0]),
            TPC(&x[0]), 
        	TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_hyb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
    	    infoA.num_cols_per_row,
   			infoA.pitch,
        	TPC(&A.matrix.ell.column_indices.values[0]),
        	TPC(&A.matrix.ell.values.values[0]),
        	TPC(&A.matrix.coo.row_indices[0]),
        	TPC(&A.matrix.coo.column_indices[0]),
        	TPC(&A.matrix.coo.values[0]),
        	TPC(&A.rs[0]),
        	TPC(&x[0]), 
    		TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_hyb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0>>> (
			infoA.num_rows,
    	    infoA.num_cols_per_row,
   			infoA.pitch,
        	TPC(&A.matrix.ell.column_indices.values[0]),
        	TPC(&A.matrix.ell.values.values[0]),
        	TPC(&A.matrix.coo.row_indices[0]),
        	TPC(&A.matrix.coo.column_indices[0]),
        	TPC(&A.matrix.coo.values[0]),
        	TPC(&A.rs[0]),
        	TPC(&x[0]), 
    		TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_csr<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
        	TPC(&A.row_offsets[0]),
        	TPC(&A.column_indices[0]),
        	TPC(&x[0]), 
    		TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_csr<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0>>> (
			infoA.num_rows,
        	TPC(&A.row_offsets[0]),
        	TPC(&A.column_indices[0]),
        	TPC(&A.values[0]),
        	TPC(&x[0]), 
    		TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_dell<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
			infoA.chunk_size,
			infoA.pitch,
			TPC(&A.Matrix_MD[0]),
			TPC(&A.ci[0]),
			TPC(&A.cl[0]),
			TPC(&A.ca[0]),
			TPC(&A.rs[0]),
			TPC(&A.cols[0]),
			TPC(&A.values[0]),
			TPC(&x[0]), 
			TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_dell<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0>>> (
			infoA.num_rows,
			infoA.chunk_size,
			infoA.pitch,
			TPC(&A.Matrix_MD[0]),
			TPC(&A.ci[0]),
			TPC(&A.cl[0]),
			TPC(&A.ca[0]),
			TPC(&A.rs[0]),
			TPC(&A.cols[0]),
			TPC(&A.values[0]),
			TPC(&x[0]), 
			TPC(&y[0]));
}

}

#endif