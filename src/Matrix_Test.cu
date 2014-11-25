#include "Matrix_Test.h"

extern void Matrix_Test(std::string filename);

void run_tests()
{
	dell_matrix_B<int, int, cusp::device_memory> DELL_mat;
	cusp::csr_matrix<int, int, cusp::device_memory> CSR_mat;
	cusp::coo_matrix<int, int, cusp::host_memory> COO_mat;

	COO_mat.resize(32, 32, 1);
	COO_mat.column_indices[0] = 1;
	COO_mat.row_indices[0] = 1;
	COO_mat.values[0] = 1;
	CSR_mat = COO_mat;

	cusp::print(CSR_mat);
	size_t rows = 32, cols = 32, csize = 16, clength = 16;
	DELL_mat.resize(rows, cols, csize, clength);
}

void SPMVTests()
{

}

void CheckMatrices(	dell_matrix_B<int, int, cusp::device_memory> &DELL_matA,
					dell_matrix_B<int, int, cusp::device_memory> &DELL_matB)
{
	mat_info<int> infoDellMatA, infoDellMatB;
	get_matrix_info<int> (DELL_matA, infoDellMatA);
	get_matrix_info<int> (DELL_matB, infoDellMatB);

	CuspVectorInt_h rows_A, rows_B, cols_A, cols_B;
	CuspVectorInt_h ciA = DELL_matA.ci;
	CuspVectorInt_h clA = DELL_matA.cl;
	CuspVectorInt_h rsA = DELL_matA.rs;
	CuspVectorInt_h ciB = DELL_matB.ci;
	CuspVectorInt_h clB = DELL_matB.cl;
	CuspVectorInt_h rsB = DELL_matB.rs;
	CuspVectorInt_h Matrix_MDA = DELL_matA.Matrix_MD;
	CuspVectorInt_h Matrix_MDB = DELL_matB.Matrix_MD;

	cols_A = DELL_matA.cols;
	cols_B = DELL_matB.cols;
	int num_rows = infoDellMatA.num_rows;
	int num_chunks = infoDellMatA.num_chunks;
	int chunk_size = infoDellMatA.chunk_size;

	int num_diff = 0;
	for(int row=0; row<num_rows; ++row)
	{
		int cID = row / chunk_size;
		int rl = rsA[row];
		int r_idx = 0;

		if(rsA[row] != rsB[row])
			fprintf(stderr, "row %d: \t DELL size: %d  HYB size: %d\n", row, rsA[row], rsB[row]);

		for(int k = 0; k < Matrix_MDA[0]; k++)
		{
			int offsetA = ciA[k*num_chunks + cID] + (row % chunk_size)*clA[k*num_chunks + cID];
			int offsetB = ciB[k*num_chunks + cID] + (row % chunk_size)*clB[k*num_chunks + cID];
			int c_idx = 0;
			
			for(; c_idx < clA[k*num_chunks + cID] && r_idx < rl; c_idx++, r_idx++)
			{
				if(cols_A[offsetA + c_idx] != cols_B[offsetB + c_idx])
				{
					fprintf(stderr, "Row: %d \t col A: %d \t col B: %d ***\n", row, cols_A[offsetA + c_idx], cols_B[offsetB + c_idx]);
					num_diff++;
				}
			}
		}
	}

	if(num_diff == 0)
		fprintf(stderr, "Matrices are identical...\n");
}

void CheckMatrices(	dell_matrix_B<int, int, cusp::device_memory> &DELL_mat,
					hyb_matrix<int, int, cusp::device_memory> &HYB_mat,
					std::vector< std::vector<int> > &GS_mat)
{
	mat_info<int> infoDellMat, infoHybMat;
	get_matrix_info<int> (DELL_mat, infoDellMat);
	get_matrix_info<int> (HYB_mat, infoHybMat);

	CuspVectorInt_h cols_A = DELL_mat.cols;
	CuspVectorInt_h cols_B = HYB_mat.matrix.ell.column_indices.values;
	CuspVectorInt_h ci = DELL_mat.ci;
	CuspVectorInt_h cl = DELL_mat.cl;
	CuspVectorInt_h ca = DELL_mat.ca;
	CuspVectorInt_h rsA = DELL_mat.rs;
	CuspVectorInt_h rsB = HYB_mat.rs;
	CuspVectorInt_h overflow_rowB = HYB_mat.matrix.coo.row_indices;
	CuspVectorInt_h overflow_colB = HYB_mat.matrix.coo.column_indices;

	int num_rows = infoDellMat.num_rows;
	int pitchA = infoDellMat.pitch;
	int pitchB = infoHybMat.pitch;
	int overflow_size = overflow_colB[0];

	std::vector< std::vector<int> > vec_mat1(GS_mat.size()), vec_mat2(GS_mat.size());
	for(int i=0; i<GS_mat.size(); ++i)
	{
		for(int j=1; j<GS_mat[i].size(); ++j)
		{
			if(GS_mat[i][j] == GS_mat[i][j-1])
				GS_mat[i].erase(GS_mat[i].begin() + j--);
		}
	}

	int num_diff = 0;
	for(int row=0; row<num_rows; ++row)
	{
		if(rsA[row] != GS_mat[row].size())
			fprintf(stderr, "*** Row Size A: %d   Row Size B: %d   Row Size GS: %d\n", rsA[row], rsB[row], GS_mat[row].size());

		int cID = row / infoDellMat.chunk_size;
		int r_idxA = 0, r_idxB = 0;

		//load DELL mat entires
		bool next_chunk = false;
		do
		{
			int offsetA = ca[cID] + (row % infoDellMat.chunk_size);
			for(int c_idx=0; c_idx < cl[cID] && r_idxA < rsA[row]; c_idx++, r_idxA++)
			{
				vec_mat1[row].push_back(cols_A[offsetA + c_idx*pitchA]);
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
		for(r_idxB=0; r_idxB < rsB[row]; r_idxB++)
		{
			vec_mat2[row].push_back(cols_B[offsetB + r_idxB*pitchB]);
		}

		for(int i=1; i <= overflow_size; i++)
		{
			if(overflow_rowB[i] == row)
				vec_mat2[row].push_back(overflow_colB[i]);
		}

		//sort vectors
		sort(vec_mat1[row].begin(), vec_mat1[row].end());
		sort(vec_mat2[row].begin(), vec_mat2[row].end());

		for(int i=0; i<GS_mat[row].size(); ++i)
		{
			//fprintf(stderr, "GS(%d, %d) \t", row, GS_mat[row][i]);
			if(vec_mat1[row][i] != GS_mat[row][i] || vec_mat2[row][i] != GS_mat[row][i])
			{
				fprintf(stderr, "A(%d, %d) \t B(%d, %d)", row, vec_mat1[row][i], row, vec_mat2[row][i]);
				num_diff++;
			}
			//fprintf(stderr, "\n");
		}
	}

	//overflow sections
	if(num_diff == 0)
		fprintf(stderr, "Matrices are identical...\n");
}

void FillTests()
{
	int mat_rows = 128;
	int mat_cols = 128;
	int N = 128;
	int num_arrays = 40;
	fprintf(stderr, "# of nonzeros: %d\n", N*num_arrays);

	int overflow_size = 4096;
	int ell_width = 32;
	int chunk_size = 32;

	dell_matrix_B<int, int, cusp::device_memory> DELL_matA;
	dell_matrix_B<int, int, cusp::device_memory> DELL_matB;
	hyb_matrix<int, int, cusp::device_memory> HYB_mat;
	DELL_matA.resize(mat_rows, mat_cols, chunk_size, ell_width);
	DELL_matB.resize(mat_rows, mat_cols, chunk_size, ell_width);
	device::Initialize_Matrix(DELL_matA);
	device::Initialize_Matrix(DELL_matB);
	HYB_mat.resize(mat_rows, mat_cols, 0, ell_width, overflow_size);

	DELL_matA.alpha = 0.5;
	DELL_matB.alpha = 0.5;

	//timer
	srand(time(NULL));
	double startTime, endTime;

	startTime = omp_get_wtime();
	CuspVectorInt_h rows_h(num_arrays*N), cols_h(num_arrays*N);
	CuspVectorInt_d rows_d(num_arrays*N), cols_d(num_arrays*N);
	std::vector< std::vector<int> > GS_mat(mat_rows);

	for(int i=0; i < num_arrays; i++)
	{
		for(int j=0; j < N; j++)
		{
			int row = rand() % mat_rows;
			int col = rand() % mat_cols;

			rows_h[i*N + j] = row;
			cols_h[i*N + j] = col;

			GS_mat[row].push_back(col);
			//fprintf(stderr, "(%d  %d)\n", rows_h[i*N + j], cols_h[i*N + j]);
		}
	}

	for(int i=0; i<mat_rows; ++i)
		sort(GS_mat[i].begin(), GS_mat[i].end());

	endTime = omp_get_wtime();
	rows_d = rows_h;
	cols_d = cols_h;
	fprintf(stderr, "gen data time: %f\n", (endTime - startTime));

	cudaPrintfInit();
	//DELL matrix
	startTime = omp_get_wtime();
	for(int i=0; i < num_arrays; i++)
	{
		device::FillMatrix(DELL_matA, rows_d, cols_d, i*N, N);
		//device::ExpandMatrix(DELL_matA);
	}
	safeSync();
	endTime = omp_get_wtime();
	cudaPrintfDisplay(stdout, true);
	fprintf(stderr, "DELL matrix load time:  %f\n", (endTime - startTime));

	//DELLW matrix
	// startTime = omp_get_wtime();
	// for(int i=0; i < num_arrays; i++)
	// {
	// 	device::FillMatrixW(DELL_matB, rows_d, cols_d, i*N, N);
	// 	//device::FillMatrix(HYB_mat, rows_d, cols_d, i*N, N);
	// }
	// safeSync();
	// endTime = omp_get_wtime();
	// cudaPrintfDisplay(stdout, true);
	// fprintf(stderr, "DELLW matrix load time:  %f\n", (endTime - startTime));

	startTime = omp_get_wtime();
	for(int i=0; i < num_arrays; i++)
	{
		device::FillMatrix(HYB_mat, rows_d, cols_d, i*N, N);
	}
	safeSync();
	endTime = omp_get_wtime();
	cudaPrintfDisplay(stdout, true);
	fprintf(stderr, "Hybrid matrix load time:  %f\n", (endTime - startTime));
	int overflow = HYB_mat.matrix.coo.column_indices[0];
	fprintf(stderr, "Overflow entries: %d\n", overflow);

	// cusp::print(DELL_matA.ci);
	// cusp::print(DELL_matA.cl);
	// cusp::print(DELL_matA.ca);

	CheckMatrices(DELL_matA, HYB_mat, GS_mat);
}


void Matrix_Test(std::string filename)
{
	FillTests();
	// run_tests();
}