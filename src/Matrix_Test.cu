#include "Matrix_Test.h"

void Matrix_Test(const std::string filename);

template <typename INDEX_TYPE, typename VALUE_TYPE>
int ReadMatrixFile(	const std::string &filename, 
					std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > &mat,
					CuspVectorInt_h &row_vec, 
					CuspVectorInt_h &col_vec,
					CuspVectorS_h &val_vec,
					int &mat_rows,
					int &mat_cols)
{
	std::ifstream mat_file(filename.c_str());
	int nnz = 0;

	if(mat_file.is_open())
	{
		int rows, cols;
		char buf[256];

		do
		{
			mat_file.getline(buf, 256);
		} while(buf[0] == '%');

		std::istringstream iss(buf);
		iss >> rows;
		iss >> cols;
		iss >> nnz;

		fprintf(stderr, "Matrix Size: %d x %d\tNNZ: %d\n", rows, cols, nnz);
		mat.resize(rows);
		row_vec.resize(nnz);
		col_vec.resize(nnz);
		val_vec.resize(nnz);

		for(int i=0; i<nnz; ++i)
		{
			int row, col;
			float val;
			mat_file.getline(buf, 256);
			std::istringstream iss(buf);

			iss >> row;
			iss >> col;
			iss >> val;
			row--;			//matrix file uses 1 based indexing
			col--;			//matrix file uses 1 based indexing
			row_vec[i] = row;
			col_vec[i] = col;
			val_vec[i] = val;
			mat[row].push_back(std::pair<int,float>(col, val));
		}

		for(int i=0; i<rows; ++i)
			sort(mat[i].begin(), mat[i].end());

		mat_rows = rows;
		mat_cols = cols;
		fprintf(stderr, "Finished reading matrix: %s\n", filename.c_str());
	}
	else
	{
		fprintf(stderr, "Error opening matrix file: %s\n", filename.c_str());
		return 0;
	}

	return nnz;
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void CheckMatrices(	dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &DELL_mat,
					hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &HYB_mat,
					const std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > &GS_mat)
{
	mat_info<int> infoDellMat, infoHybMat;
	get_matrix_info<int> (DELL_mat, infoDellMat);
	get_matrix_info<int> (HYB_mat, infoHybMat);

	CuspVectorInt_h cols_A = DELL_mat.cols;
	CuspVectorInt_h cols_B = HYB_mat.matrix.ell.column_indices.values;
	CuspVectorS_h vals_A = DELL_mat.values;
	CuspVectorS_h vals_B = HYB_mat.matrix.ell.values.values;
	CuspVectorInt_h ci = DELL_mat.ci;
	CuspVectorInt_h cl = DELL_mat.cl;
	CuspVectorInt_h ca = DELL_mat.ca;
	CuspVectorInt_h rsA = DELL_mat.rs;
	CuspVectorInt_h rsB = HYB_mat.rs;
	CuspVectorInt_h overflow_rowB = HYB_mat.matrix.coo.row_indices;
	CuspVectorInt_h overflow_colB = HYB_mat.matrix.coo.column_indices;
	CuspVectorS_h overflow_valsB = HYB_mat.matrix.coo.values;

	int num_rows = infoDellMat.num_rows;
	int pitchA = infoDellMat.pitch;
	int pitchB = infoHybMat.pitch;
	int ell_size = infoHybMat.num_cols_per_row;
	int overflow_size = overflow_colB[0];

	std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > vec_mat1(GS_mat.size()), vec_mat2(GS_mat.size());

	int nnz = 0;
	int num_diff = 0;
	for(int row=0; row<num_rows; ++row)
	{
		int cID = row / infoDellMat.chunk_size;
		int r_idxA = 0, r_idxB = 0;

		//load DELL mat entires
		bool next_chunk = false;
		do
		{
			//int offsetA = ca[cID] + (row % infoDellMat.chunk_size);			//column aligned memory accesses
			int offsetA = ca[cID] + (row % infoDellMat.chunk_size)*pitchA;		//row aligned memory accesses
			for(int c_idx=0; c_idx < cl[cID] && r_idxA < rsA[row]; c_idx++, r_idxA++)
			{
				//vec_mat1[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols_A[offsetA + c_idx*pitchA], vals_A[offsetA + c_idx*pitchA]) );
				vec_mat1[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols_A[offsetA + c_idx], vals_A[offsetA + c_idx]) );
			}

			if(ci[cID] > 0)
			{
				next_chunk = true;
				cID = ci[cID];
			}
			else
				next_chunk = false;

		} while(next_chunk);

		//load HYB mat entries
		int offsetB = row;
		for(r_idxB=0; r_idxB < rsB[row] && r_idxB < ell_size; r_idxB++)
		{
			vec_mat2[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols_B[offsetB + r_idxB*pitchB], vals_B[offsetB + r_idxB*pitchB]) );
		}

		for(int i=1; i <= overflow_size; i++)
		{
			if(overflow_rowB[i] == row)
				vec_mat2[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(overflow_colB[i], overflow_valsB[i]) );
		}

		//sort vectors
		sort(vec_mat1[row].begin(), vec_mat1[row].end());
		sort(vec_mat2[row].begin(), vec_mat2[row].end());

		if(vec_mat1[row].size() != GS_mat[row].size())
				fprintf(stderr, "*** Row Size A: %d   Row Size GS: %d\n", vec_mat1[row].size(), GS_mat[row].size());

		if(vec_mat1[row].size() != vec_mat2[row].size())
			fprintf(stderr, "*** Row %d \t A Size: %d \t B Size: %d\n", row, vec_mat1[row].size(), vec_mat2[row].size());

		for(int i=0; i<GS_mat[row].size(); ++i)
		{
			if(	vec_mat1[row][i].first != GS_mat[row][i].first || vec_mat2[row][i].first != GS_mat[row][i].first ||
				vec_mat1[row][i].second != GS_mat[row][i].second || vec_mat2[row][i].second != GS_mat[row][i].second)
			{
				fprintf(stderr, "GS(%d, %d):  %f\t", row, GS_mat[row][i].first, GS_mat[row][i].second);
				fprintf(stderr, "DELL(%d, %d):  %f \t HYB(%d, %d):  %f", row, vec_mat1[row][i].first, vec_mat1[row][i].second, row, vec_mat2[row][i].first, vec_mat2[row][i].second);
				fprintf(stderr, "\n");
				num_diff++;
			}
		}

		nnz += GS_mat[row].size();
	}

	//overflow sections
	if(num_diff == 0)
		fprintf(stderr, "Matrices are identical...\n");
	else
		fprintf(stderr, "Matrices have %d differences...\n", num_diff);

	fprintf(stderr, "Number of Nonzeros in final matrix: %d\n", nnz);
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void CheckMatrices(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &CSR_mat,
					const std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > &GS_mat)
{
	mat_info<int> infoCSRMat;
	get_matrix_info<int> (CSR_mat, infoCSRMat);

	CuspVectorInt_h row_offsets = CSR_mat.row_offsets;
	CuspVectorInt_h cols = CSR_mat.column_indices;
	CuspVectorS_h vals = CSR_mat.values;

	int num_rows = infoCSRMat.num_rows;
	std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > vec_mat1(GS_mat.size());

	int nnz = 0;
	int num_diff = 0;
	for(int row=0; row<num_rows; ++row)
	{
		for(int idx=row_offsets[row]; idx < row_offsets[row+1]; ++idx)
		{
			vec_mat1[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols[idx], vals[idx]) );
		}

		//sort vectors
		sort(vec_mat1[row].begin(), vec_mat1[row].end());

		if(vec_mat1[row].size() != GS_mat[row].size())
				fprintf(stderr, "*** Row Size A: %d   Row Size GS: %d\n", vec_mat1[row].size(), GS_mat[row].size());

		for(int i=0; i<GS_mat[row].size(); ++i)
		{
			if(	vec_mat1[row][i].first != GS_mat[row][i].first || vec_mat1[row][i].second != GS_mat[row][i].second )
			{
				fprintf(stderr, "GS(%d, %d):  %f\t", row, GS_mat[row][i].first, GS_mat[row][i].second);
				fprintf(stderr, "CSR(%d, %d):  %f", row, vec_mat1[row][i].first, vec_mat1[row][i].second);
				fprintf(stderr, "\n");
				num_diff++;
			}
		}

		nnz += GS_mat[row].size();
	}

	//overflow sections
	if(num_diff == 0)
		fprintf(stderr, "Matrices are identical...\n");
	else
		fprintf(stderr, "Matrices have %d differences...\n", num_diff);

	fprintf(stderr, "Number of Nonzeros in final matrix: %d\n", nnz);
}

void FillTests(const std::string &filename)
{
	int mat_rows, mat_cols;
	CuspVectorInt_h rows_h, cols_h;
	CuspVectorInt_d rows_d, cols_d;
	CuspVectorS_h vals_h, x_vec_h, y1_vec_h, y2_vec_h, y3_vec_h;
	CuspVectorS_d vals_d, x_vec_d, y_vec_d;

	std::vector< std::vector< std::pair<int,float> > > GS_mat;
	int NNZ = ReadMatrixFile(filename, GS_mat, rows_h, cols_h, vals_h, mat_rows, mat_cols);
	if(NNZ == 0)
		exit(1);
	
	//calcuate average 
	int max_rows = 0, nnz_count = 0, avg_row, est_ovf = 0;
	float std_dev = 0;
	for(int i=0; i<GS_mat.size(); ++i)
	{
		if(GS_mat[i].size() > max_rows)
			max_rows = GS_mat[i].size();
		nnz_count += GS_mat[i].size();

		if(GS_mat[i].size() > 16)
			est_ovf += GS_mat[i].size() - 16;
	}
	avg_row = nnz_count / mat_rows;
	//caculate standard deviation
	for(int i=0; i<GS_mat.size(); ++i)
	{
		std_dev += (GS_mat[i].size() - avg_row) * (GS_mat[i].size() - avg_row);
	}
	std_dev = sqrt(std_dev / mat_rows);
	fprintf(stderr, "average entries per row: %d\n", avg_row);
	fprintf(stderr, "max row size: %d\n", max_rows);
	fprintf(stderr, "standard deviation: %f\n", std_dev);
	fprintf(stderr, "nnz count: %d\tNNZ returned: %d\n", nnz_count, NNZ);
	fprintf(stderr, "estimated overflow for Hybrid matrix with ell width of 16: %d\n", est_ovf);

	int num_arrays = 1;
	int ell_width = (avg_row / 16 + ((avg_row % 16 == 0) ? 0 : 1)) * 16;
	int chunk_size = 16;
	int overflow_size = mat_rows*ell_width*7;

	cusp::csr_matrix<int, float, cusp::device_memory> CSR_mat;
	CSR_mat.resize(mat_rows, mat_cols, NNZ);
	hyb_matrix<int, float, cusp::device_memory> HYB_mat;
	HYB_mat.resize(mat_rows, mat_cols, overflow_size, ell_width);
	dell_matrix<int, float, cusp::device_memory> DELL_matA;
	DELL_matA.resize(mat_rows, mat_cols, chunk_size, ell_width);
	device::Initialize_Matrix(DELL_matA);
	DELL_matA.alpha = 2.0;

	fprintf(stderr, "rows: %d  cols: %d\n", mat_rows, mat_cols);
	fprintf(stderr, "ell width: %d\n", ell_width);
	fprintf(stderr, "chunk_size: %d\n", chunk_size);
	fprintf(stderr, "chunk_length: %d\n", ell_width);
	fprintf(stderr, "overflow memsize: %d\n", overflow_size);

	//timer
	srand(time(NULL));
	double startTime, endTime;
	
	//generate random vector
	x_vec_h.resize(mat_cols);
	for(int i=0; i<mat_cols; ++i)
		x_vec_h[i] = rand() % 10;

	x_vec_d = x_vec_h;
	rows_d = rows_h;
	cols_d = cols_h;
	vals_d = vals_h;

	cudaPrintfInit();
	//DELL matrix
	startTime = omp_get_wtime();
	for(int i=0; i < num_arrays; i++)
	{
		device::LoadMatrix(DELL_matA, rows_d, cols_d, vals_d, NNZ);
	}
	safeSync();
	endTime = omp_get_wtime();
	cudaPrintfDisplay(stdout, true);
	fprintf(stderr, "DELL matrix load time:  %f\n", (endTime - startTime));

	startTime = omp_get_wtime();
	for(int i=0; i < num_arrays; i++)
	{
		device::LoadMatrix(HYB_mat, rows_d, cols_d, vals_d, NNZ);
	}
	safeSync();
	endTime = omp_get_wtime();
	cudaPrintfDisplay(stdout, true);
	fprintf(stderr, "Hybrid matrix load time:  %f\n", (endTime - startTime));
	int overflow = HYB_mat.matrix.coo.column_indices[0];
	fprintf(stderr, "Overflow entries: %d\n", overflow);

	//CSR matrix
	startTime = omp_get_wtime();
	for(int i=0; i < num_arrays; i++)
	{
		//device::ConvertMatrix(DELL_matA, CSR_mat);
		device::LoadMatrix(CSR_mat, rows_d, cols_d, vals_d, NNZ);
	}
	safeSync();
	endTime = omp_get_wtime();
	cudaPrintfDisplay(stdout, true);
	fprintf(stderr, "CSR matrix load time:  %f\n", (endTime - startTime));

	// cusp::print(DELL_matA.ci);
	// cusp::print(DELL_matA.cl);
	// cusp::print(DELL_matA.ca);

	fprintf(stderr, "test\n");
	//Check to ensure that matrices are equivalent to CPU generated matrix
	CheckMatrices(DELL_matA, HYB_mat, GS_mat);
	CheckMatrices(CSR_mat, GS_mat);

	//SPMV tests
	#define TEST_COUNT	500

	y_vec_d.resize(mat_cols, 0);
	startTime = omp_get_wtime();
	for(int i=0; i<TEST_COUNT; i++)
	{
		device::spmv(DELL_matA, x_vec_d, y_vec_d);
	}
	safeSync();
	endTime = omp_get_wtime();
	fprintf(stderr, "DELL matrix SpMV time:  %f\n", (endTime - startTime));
	y1_vec_h = y_vec_d;

	y_vec_d.resize(mat_cols, 0);
	startTime = omp_get_wtime();
	for(int i=0; i<TEST_COUNT; i++)
	{
		device::spmv(HYB_mat, x_vec_d, y_vec_d);
	}
	safeSync();
	endTime = omp_get_wtime();
	fprintf(stderr, "Hybrid matrix SpMV time:  %f\n", (endTime - startTime));
	y2_vec_h = y_vec_d;

	y_vec_d.resize(mat_cols, 0);
	startTime = omp_get_wtime();
	for(int i=0; i<TEST_COUNT; i++)
	{
		device::spmv(CSR_mat, x_vec_d, y_vec_d);
	}
	safeSync();
	endTime = omp_get_wtime();
	fprintf(stderr, "CSR matrix SpMV time:  %f\n", (endTime - startTime));
	y3_vec_h = y_vec_d;

	for(int i=0; i<mat_cols; ++i)
	{
		if(fabs(y1_vec_h[i] - y2_vec_h[i]) > 1e-9 || fabs(y1_vec_h[i] - y3_vec_h[i]) > 1e-9 || fabs(y2_vec_h[i] - y3_vec_h[i]) > 1e-9)
		{
			float sum = 0;
			for(int j=0; j<GS_mat[i].size(); j++)
				sum += GS_mat[i][j].second * x_vec_h[GS_mat[i][j].first];

			fprintf(stderr, "ERROR   %d:  %f\t%f\t%f\t%f\n", i, y1_vec_h[i], y2_vec_h[i], y3_vec_h[i], sum);
		}
		//fprintf(stderr, "row %d:  %f\t%f\n", i, y1_vec_h[i], y2_vec_h[i]);
	}
}


void Matrix_Test(const std::string filename)
{
	FillTests(filename);
	// run_tests();
}