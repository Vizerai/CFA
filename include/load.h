#ifndef LOAD_H
#define LOAD_H

namespace device
{

//initialize DELL matrix
template <typename INDEX_TYPE, typename VALUE_TYPE>
void Initialize_Matrix(	dell_matrix_B<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	InitializeMatrix_dell<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_chunks,
			infoMat.chunk_size,
			infoMat.chunk_length,
			infoMat.pitch,
			TPC(&mat.Matrix_MD[0]),
			TPC(&mat.ci[0]),
			TPC(&mat.cl[0]),
			TPC(&mat.ca[0]),
			TPC(&mat.rs[0]));
}

//initialize DELL matrix
template <typename INDEX_TYPE, typename VALUE_TYPE>
void Initialize_Matrix(	dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	InitializeMatrix_dell<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_chunks,
			infoMat.chunk_size,
			infoMat.chunk_length,
			infoMat.pitch,
			TPC(&mat.Matrix_MD[0]),
			TPC(&mat.ci[0]),
			TPC(&mat.cl[0]),
			TPC(&mat.ca[0]),
			TPC(&mat.rs[0]));
}

//*******************************************************************************************//
//Fill matrices from a COO matrix
//*******************************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	dell_matrix_B<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	LoadMatrix_dell_B_coo<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.chunk_size,
			infoMat.pitch,
			TPC(&rows[0]),
			TPC(&cols[0]),
			N,
			TPC(&mat.Matrix_MD[0]),
			TPC(&mat.ci[0]),
			TPC(&mat.cl[0]),
			TPC(&mat.ca[0]),
			TPC(&mat.rs[0]),
			TPC(&mat.cols[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	LoadMatrix_dell_coo<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.chunk_size,
			infoMat.pitch,
			infoMat.alpha,
			TPC(&rows[0]),
			TPC(&cols[0]),
			TPC(&vals[0]),
			N,
			TPC(&mat.Matrix_MD[0]),
			TPC(&mat.ci[0]),
			TPC(&mat.cl[0]),
			TPC(&mat.ca[0]),
			TPC(&mat.rs[0]),
			TPC(&mat.cols[0]),
			TPC(&mat.values[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

#if(DEBUG)
	assert(src.num_rows == infoDst.num_rows);
	assert(src.num_cols == infoDst.num_cols);
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	LoadMatrix_hyb_B_coo<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_cols,
			infoMat.num_cols_per_row,
			infoMat.pitch,
			TPC(&rows[0]),
			TPC(&cols[0]),
			N,
			TPC(&mat.rs[0]),
			TPC(&mat.matrix.ell.column_indices.values[0]),
			TPC(&mat.matrix.coo.row_indices[0]),
			TPC(&mat.matrix.coo.column_indices[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

#if(DEBUG)
	assert(src.num_rows == infoDst.num_rows);
	assert(src.num_cols == infoDst.num_cols);
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	LoadMatrix_hyb_coo<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_cols,
			infoMat.num_cols_per_row,
			infoMat.pitch,
			TPC(&rows[0]),
			TPC(&cols[0]),
			TPC(&vals[0]),
			N,
			TPC(&mat.rs[0]),
			TPC(&mat.matrix.ell.column_indices.values[0]),
			TPC(&mat.matrix.ell.values.values[0]),
			TPC(&mat.matrix.coo.row_indices[0]),
			TPC(&mat.matrix.coo.column_indices[0]),
			TPC(&mat.matrix.coo.values[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

#if(DEBUG)
	assert(src.num_rows == infoDst.num_rows);
	assert(src.num_cols == infoDst.num_cols);
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	//perform a scan on the entries to determine row lengths and update the row vector
	LoadMatrix_csr_coo_SCAN<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_cols,
			TPC(&rows[0]),
			N,
			TPC(&mat.row_offsets[0]));

	thrust::exclusive_scan(mat.row_offsets.begin(), mat.row_offsets.end(), mat.row_offsets.begin());

	LoadMatrix_csr_coo<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_cols,
			TPC(&rows[0]),
			TPC(&cols[0]),
			TPC(&vals[0]),
			N,
			TPC(&mat.row_offsets[0]),
			TPC(&mat.column_indices[0]),
			TPC(&mat.values[0]));
}

//*******************************************************************************************//
//Load matrices from a CSR matrix
//*******************************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
					cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
{
	dst.resize(src.num_rows, src.num_cols, src.num_entries, std::max(src.num_cols/16, ulong(64)));

	mat_info<INDEX_TYPE> infoDst;
	get_matrix_info(dst, infoDst);

#if(DEBUG)
	assert(src.num_rows == infoDst.num_rows);
	assert(src.num_cols == infoDst.num_cols);
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	LoadEllMatrix<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			src.num_rows,
			src.num_entries,
			infoDst.num_cols_per_row,
			infoDst.pitch,
			TPC(&src.row_offsets[0]),
			TPC(&src.column_indices[0]),
			TPC(&src.values[0]),
			TPC(&dst.column_indices.values[0]),
			TPC(&dst.values.values[0]));

}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
					hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
{
	//src.sort_by_row_and_column();
	const INDEX_TYPE invalid_index = -1;
	dst.resize(src.num_rows, src.num_cols, src.num_entries, 256, std::max(src.num_cols/16, ulong(96)));
	thrust::fill(dst.matrix.ell.column_indices.values.begin(), dst.matrix.ell.column_indices.values.end(), invalid_index);

	mat_info<INDEX_TYPE> infoDst;
	get_matrix_info(dst, infoDst);

#if(DEBUG)
	assert(src.num_rows == infoDst.num_rows);
	assert(src.num_cols == infoDst.num_cols);
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	LoadHybMatrix<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			src.num_rows,
			src.num_entries,
			infoDst.num_cols_per_row,
			infoDst.pitch,
			TPC(&src.row_offsets[0]),
			TPC(&src.column_indices[0]),
			TPC(&dst.row_sizes[0]),
			TPC(&dst.matrix.ell.column_indices.values[0]),
			TPC(&dst.matrix.coo.row_indices[0]),
			TPC(&dst.matrix.coo.column_indices[0]));

	dst.matrix.num_entries = src.num_entries;
}

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
// 					dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
// {
// 	#define ROW_SIZE	1024
// 	//dst.resize(src.num_rows, src.num_cols, src.num_entries, src.num_rows*ROW_SIZE);

// 	mat_info<INDEX_TYPE> infoDst;
// 	get_matrix_info(dst, infoDst);

// #if(DEBUG)
// 	assert(src.num_rows == infoDst.num_rows);
// 	assert(src.num_cols == infoDst.num_cols);
// #endif

// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREADS;

// 	// LoadCSRMatrix<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>>

// 	dst.num_entries = src.num_entries;
// }

//*******************************************************************************************//
//Fill matrices from a COO format
//*******************************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE>
void UpdateMatrix(	dell_matrix_B<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	UpdateMatrix_dell_B<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.chunk_size,
			infoMat.pitch,
			infoMat.alpha,
			TPC(&rows[0]),
			TPC(&cols[0]),
			N,
			TPC(&mat.Matrix_MD[0]),
			TPC(&mat.ci[0]),
			TPC(&mat.cl[0]),
			TPC(&mat.ca[0]),
			TPC(&mat.rs[0]),
			TPC(&mat.cols[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void UpdateMatrix(	dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	UpdateMatrix_dell<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.chunk_size,
			infoMat.pitch,
			infoMat.alpha,
			TPC(&rows[0]),
			TPC(&cols[0]),
			TPC(&vals[0]),
			N,
			TPC(&mat.Matrix_MD[0]),
			TPC(&mat.ci[0]),
			TPC(&mat.cl[0]),
			TPC(&mat.ca[0]),
			TPC(&mat.rs[0]),
			TPC(&mat.cols[0]),
			TPC(&mat.values[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void UpdateMatrix(	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

#if(DEBUG)
	assert(src.num_rows == infoDst.num_rows);
	assert(src.num_cols == infoDst.num_cols);
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	UpdateMatrix_hyb_B<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_cols,
			infoMat.num_cols_per_row,
			infoMat.pitch,
			TPC(&rows[0]),
			TPC(&cols[0]),
			N,
			TPC(&mat.rs[0]),
			TPC(&mat.matrix.ell.column_indices.values[0]),
			TPC(&mat.matrix.coo.row_indices[0]),
			TPC(&mat.matrix.coo.column_indices[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void UpdateMatrix(	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

#if(DEBUG)
	assert(src.num_rows == infoDst.num_rows);
	assert(src.num_cols == infoDst.num_cols);
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	UpdateMatrix_hyb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_cols,
			infoMat.num_cols_per_row,
			infoMat.pitch,
			TPC(&rows[0]),
			TPC(&cols[0]),
			TPC(&vals[0]),
			N,
			TPC(&mat.rs[0]),
			TPC(&mat.matrix.ell.column_indices.values[0]),
			TPC(&mat.matrix.ell.values.values[0]),
			TPC(&mat.matrix.coo.row_indices[0]),
			TPC(&mat.matrix.coo.column_indices[0]),
			TPC(&mat.matrix.coo.values[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void ConvertMatrix(	const dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
					cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
{
	mat_info<INDEX_TYPE> infoSrc, infoDst;
	get_matrix_info(src, infoSrc);
	get_matrix_info(dst, infoDst);

#if(DEBUG)
	assert(src.num_rows == infoDst.num_rows);
	assert(src.num_cols == infoDst.num_cols);
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	thrust::copy_n(src.rs.begin(), infoSrc.num_rows, dst.row_offsets.begin());
	thrust::exclusive_scan(dst.row_offsets.begin(), dst.row_offsets.end(), dst.row_offsets.begin());

	// ConvertMatrix_DELL_CSR<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
	// 		infoSrc.num_rows,
	// 		infoSrc.chunk_size,
	// 		infoSrc.pitch,
	// 		TPC(&src.Matrix_MD[0]),
	// 		TPC(&src.ci[0]),
	// 		TPC(&src.cl[0]),
	// 		TPC(&src.ca[0]),
	// 		TPC(&src.rs[0]),
	// 		TPC(&src.cols[0]),
	// 		TPC(&src.values[0]),
	// 		TPC(&dst.row_offsets[0]),
	// 		TPC(&dst.column_indices[0]),
	// 		TPC(&dst.values[0]));
}

} //namespace device

#endif