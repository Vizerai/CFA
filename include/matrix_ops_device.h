#ifndef MATRIX_OPS_DEVICE
#define MATRIX_OPS_DEVICE

<<<<<<< HEAD
#define QUEUE_SIZE		512

namespace device
{

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
=======
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
__global__ void AccumMat(	const 	INDEX_TYPE num_rows,
							const 	INDEX_TYPE num_cols,
							const 	INDEX_TYPE num_cols_per_row,
							const 	INDEX_TYPE pitch,
							const 	INDEX_TYPE *column_indices,
									VALUE_TYPE *a)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	const INDEX_TYPE invalid_index = cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory>::invalid_index;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE offset = row;
		for(int n=0; n < num_cols_per_row; ++n)
		{
			int col = column_indices[offset];
			if(col != invalid_index)
			{
				a[row] = 1;
			}
			offset += pitch;
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
<<<<<<< HEAD
__launch_bounds__(BLOCK_THREADS_MAX,1)
=======
__launch_bounds__(BLOCK_THREADS,1)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
__global__ void 
column_select(	const INDEX_TYPE num_rows,
				const INDEX_TYPE *A_row_offsets,
				const INDEX_TYPE *A_column_indices,
				const VALUE_TYPE *s,
				VALUE_TYPE *y)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	const INDEX_TYPE index = s[0];

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE offset = A_row_offsets[row];
		INDEX_TYPE num_cols_per_row = A_row_offsets[row + 1] - offset;

		VALUE_TYPE val = 0;
		for(int n=0; n < num_cols_per_row; ++n, ++offset)
		{
			int col = A_column_indices[offset];
			if(col == index)
			{
				val = 1;
			}
		}

		y[row] = val;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
<<<<<<< HEAD
__launch_bounds__(BLOCK_THREADS_MAX,1)
=======
__launch_bounds__(BLOCK_THREADS,1)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
__global__ void 
column_select_if(	const INDEX_TYPE num_rows,
					const INDEX_TYPE *A_row_offsets,
					const INDEX_TYPE *A_column_indices,
					const VALUE_TYPE *s,
					const VALUE_TYPE *cond,
					VALUE_TYPE *y)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	//check conditional value
	if(cond[0] == 0)
		return;

	const int index = s[0];

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE offset = A_row_offsets[row];
		INDEX_TYPE num_cols_per_row = A_row_offsets[row + 1] - offset;

		VALUE_TYPE val = 0;
		for(int n=0; n < num_cols_per_row; ++n, ++offset)
		{
			int col = A_column_indices[offset];
			if(col == index)
			{
				val = 1;
			}
		}

		y[row] = val;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
<<<<<<< HEAD
__launch_bounds__(BLOCK_THREADS_MAX,1)
=======
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
OuterProduct(	const VALUE_TYPE *a,
				const VALUE_TYPE *b,
				const INDEX_TYPE num_rows,
				const INDEX_TYPE num_cols,
				const INDEX_TYPE num_cols_per_row,
				const INDEX_TYPE pitch,
				INDEX_TYPE *column_indices,
				VALUE_TYPE *values)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	
	const INDEX_TYPE invalid_index = cusp::ell_matrix<int, INDEX_TYPE, cusp::device_memory>::invalid_index;

	__shared__ int entries_b[BLOCK_THREADS];
	__shared__ int num_entries_a, num_entries_b;

	entries_b[threadIdx.x] = -1;
	__syncthreads();

	if(threadIdx.x == 0)		//first thread of every block
	{
		num_entries_a = 0;
		num_entries_b = 0;

		for(int i=0; i < num_rows; ++i)
		{
			if(a[i] != 0)
				num_entries_a++;
		}

		for(int i=0; i < num_cols; ++i)
		{
			if(b[i] != 0)
			{
				entries_b[num_entries_b] = i;
				num_entries_b++;
			}
		}
	}
	__syncthreads();

	for(int row=tID; row<num_rows; row+=grid_size)
	{
		int offset = row;
		if(a[row])
		{
			for(int n=0; n < num_entries_b; ++n, offset+=pitch)
			{
				column_indices[offset] = entries_b[n];
				values[offset] = 1;
			}
		}

		while(offset < num_cols_per_row*pitch)
		{
			column_indices[offset] = invalid_index;
			offset += pitch;
		}
	}
}

//size of a must be num_rows + 1
//size of b must be num_cols + 1
//last entry of each array is used for storing entry count
//only add unique entries
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
OuterProductAdd_ELL(	const VALUE_TYPE *a,
						const VALUE_TYPE *b,
						const INDEX_TYPE *index_count,
						const INDEX_TYPE num_rows,
						const INDEX_TYPE num_cols,
						const INDEX_TYPE num_cols_per_row,
						const INDEX_TYPE pitch,
						INDEX_TYPE *column_indices,
						VALUE_TYPE *values)
{
	const int tID = threadIdx.x & (WARP_SIZE-1); 									//thread ID
	const int wID = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;			//warp ID
	const int grid_size = (blockDim.x * gridDim.x) / WARP_SIZE;
	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory>::invalid_index;

	INDEX_TYPE num_entries_a = index_count[0];
	INDEX_TYPE num_entries_b = index_count[1];
	__shared__ INDEX_TYPE row_index[BLOCK_THREADS/WARP_SIZE];

	for(INDEX_TYPE j=wID; j < num_entries_a; j+=grid_size)
	{
		INDEX_TYPE row = a[j];
		row_index[wID] = column_indices[row];
		for(INDEX_TYPE k=tID; k < num_entries_b; k+=WARP_SIZE)
		{
			VALUE_TYPE b_col = b[k];
			INDEX_TYPE offset = row;
			for(INDEX_TYPE n=1; n < num_cols_per_row; ++n, offset+=pitch)
			{
				INDEX_TYPE col = column_indices[offset];
				if(col == b_col)
				{
					break;
				}
				else if(col == invalid_index)
				{
					column_indices[row*(atomicAdd(&row_index[wID],1)+1)] = b_col;
					//values[offset] = 1;
					break;
				}
			}
		}
		if(tID == 0)
			column_indices[row] = row_index[wID];
	}
}

//size of a must be num_rows + 1
//size of b must be num_cols + 1
//last entry of each array is used for storing entry count
//only add unique entries
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
OuterProductAdd_HYB(	const VALUE_TYPE *a,
						const VALUE_TYPE *b,
						const INDEX_TYPE *index_count,
						const INDEX_TYPE num_rows,
						const INDEX_TYPE num_cols,
						const INDEX_TYPE num_cols_per_row,
						const INDEX_TYPE pitch,
						INDEX_TYPE *row_sizes,
						INDEX_TYPE *column_indices,
						INDEX_TYPE *coo_row_indices,
						INDEX_TYPE *coo_column_indices)
{
	//const int tID = threadIdx.x & (WARP_SIZE-1); 									//thread ID
	//const int wID = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;			//warp ID
	//const int grid_size = (blockDim.x * gridDim.x) / WARP_SIZE;
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory>::invalid_index;

	INDEX_TYPE num_entries_a = index_count[0];
	INDEX_TYPE num_entries_b = index_count[1];

	//for(INDEX_TYPE j=wID; j < num_entries_a; j+=grid_size)
	for(INDEX_TYPE j=tID; j < num_entries_a; j+=grid_size)
	{
		INDEX_TYPE row = a[j];
		//for(INDEX_TYPE k=tID; k < num_entries_b; k+=WARP_SIZE)
		for(INDEX_TYPE k=0; k<num_entries_b; ++k)
		{
			VALUE_TYPE b_col = b[k];
			INDEX_TYPE offset = row+pitch;
			bool overflow = false;
			for(INDEX_TYPE n=1; n < num_cols_per_row; ++n, offset+=pitch)
			{
				INDEX_TYPE col = column_indices[offset];
				if(col == b_col)
				{
					if(n == num_cols_per_row-1)
						overflow = true;
					break;
				}
				else if(col == invalid_index)
				{
					INDEX_TYPE index = atomicAdd(&row_sizes[row], 1) + 1;
					column_indices[row + pitch*index] = b_col;
					break;
				}
			}

			//coordinate overflow
			if(overflow)
			{
				bool valid = true;
				for(int i=1; i < coo_column_indices[0]; ++i)
				{
					if(coo_column_indices[i] == b_col && coo_row_indices[i] == row)
					{
						valid = false;
						break;
					}
				}

				if(valid)
				{
					int index = atomicAdd(&coo_column_indices[0], 1)+1;
					coo_row_indices[index] = row;
					coo_column_indices[index] = b_col;
				}
			}
		}
	}
}

//size of a must be num_rows + 1
//size of b must be num_cols + 1
//last entry of each array is used for storing entry count
//only add unique entries
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
OuterProductAdd_DELL(	const VALUE_TYPE *a,
						const VALUE_TYPE *b,
						const INDEX_TYPE *index_count,
						const INDEX_TYPE *row_offsets,
						INDEX_TYPE *column_indices,
						INDEX_TYPE *row_sizes,
						INDEX_TYPE *coo_row_indices,
						INDEX_TYPE *coo_column_indices)
{
	//const int tID = threadIdx.x & (WARP_SIZE-1); 									//thread ID
	//const int wID = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;			//warp ID
	//const int grid_size = (blockDim.x * gridDim.x) / WARP_SIZE;
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = -1;

	INDEX_TYPE num_entries_a = index_count[0];
	INDEX_TYPE num_entries_b = index_count[1];

	//for(INDEX_TYPE j=wID; j < num_entries_a; j+=grid_size)
	for(INDEX_TYPE j=tID; j < num_entries_a; j+=grid_size)
	{
		INDEX_TYPE row = a[j];
		//for(INDEX_TYPE k=tID; k < num_entries_b; k+=WARP_SIZE)
		for(INDEX_TYPE k=0; k<num_entries_b; ++k)
		{
			VALUE_TYPE b_col = b[k];
			INDEX_TYPE row_start = row_offsets[row];
			INDEX_TYPE row_end = row_offsets[row+1];
			
			bool overflow = true;
			for(INDEX_TYPE n=row_start; n < row_end; ++n)
			{
				INDEX_TYPE col = column_indices[n];
				if(col == b_col)
				{
					overflow = false;
					break;
				}
				else if(col == invalid_index)
				{
					if(atomicCAS(&column_indices[n], invalid_index, b_col) == -1)
					{
						atomicAdd(&row_sizes[row], 1);	
						overflow = false;
						break;
					}
					else
						n--;
				}
			}

			//coordinate overflow
			if(overflow)
			{
				bool valid = true;
				for(int i=1; i < coo_column_indices[0]+1; ++i)
				{
					if(coo_column_indices[i] == b_col && coo_row_indices[i] == row)
					{
						valid = false;
						break;
					}
				}

				if(valid)
				{
					int index = coo_column_indices[0]+1;
					if(atomicCAS(&coo_column_indices[index], -1, b_col) == -1)
					{
						atomicAdd(&coo_column_indices[0], 1);
						coo_row_indices[index] = row;
						coo_column_indices[index] = b_col;
					}
				}
			}
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void
spmv_ellb(	const INDEX_TYPE num_rows,
            const INDEX_TYPE num_cols_per_row,
            const INDEX_TYPE pitch,
            const INDEX_TYPE * Aj,
            const VALUE_TYPE * x, 
                  VALUE_TYPE * y)
{
    const INDEX_TYPE invalid_index = cusp::ell_matrix<int, INDEX_TYPE, cusp::device_memory>::invalid_index;

    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
    	VALUE_TYPE sum = 0;
        INDEX_TYPE offset = row;
        for(INDEX_TYPE n = 0; n < num_cols_per_row; n++)
        {
            const INDEX_TYPE col = Aj[offset];
            if(col != invalid_index)
            {
	            if(x[col] != 0)
	            	sum = 1;
            }

            offset += pitch;
        }

        y[row] = sum;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void
spmv_hybb(	const INDEX_TYPE num_rows,
            const INDEX_TYPE num_cols_per_row,
            const INDEX_TYPE pitch,
            const INDEX_TYPE * A_ell_column_indices,
        	const INDEX_TYPE * A_coo_row_indices,
            const INDEX_TYPE * A_coo_column_indices,
            const VALUE_TYPE * x, 
                  VALUE_TYPE * y)
{
    const INDEX_TYPE invalid_index = cusp::ell_matrix<int, INDEX_TYPE, cusp::device_memory>::invalid_index;

    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
    	VALUE_TYPE sum = 0;
        INDEX_TYPE offset = row+pitch;
        for(INDEX_TYPE n = 0; n < num_cols_per_row; n++)
        {
            const INDEX_TYPE col = A_ell_column_indices[offset];
            if(col != invalid_index)
            {
	            if(x[col] != 0)
	            	sum = 1;
            }
            else
            	break;

            offset += pitch;
        }

        for(int n=1; n < A_coo_column_indices[0]; n++)
        {
        	if(A_coo_row_indices[n] == row)
        	{
        		const INDEX_TYPE col = A_coo_column_indices[n];
        		if(x[col] != 0)
	            	sum = 1;
        	}
        }

        y[row] = sum;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void
spmv_csrb(	const INDEX_TYPE num_rows,
            const INDEX_TYPE * A_row_offsets,
			const INDEX_TYPE * A_column_indices,
            const VALUE_TYPE * x, 
                  VALUE_TYPE * y)
{
    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
    	INDEX_TYPE row_start = A_row_offsets[row];
    	INDEX_TYPE row_end = A_row_offsets[row + 1];

    	VALUE_TYPE sum = 0;
        for(INDEX_TYPE j=row_start; j < row_end; ++j)
        {
        	INDEX_TYPE col = A_column_indices[j];
        	if(x[col] != 0)
	    		sum = 1;
        }

        y[row] = sum;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void
spmv_dellb(	const INDEX_TYPE num_rows,
            const INDEX_TYPE * A_row_offsets,
			const INDEX_TYPE * A_column_indices,
			const INDEX_TYPE * A_coo_row_indices,
			const INDEX_TYPE * A_coo_column_indices,
            const VALUE_TYPE * x, 
                  VALUE_TYPE * y)
{
    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;
    const INDEX_TYPE coo_end = A_coo_column_indices[0];

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
    	INDEX_TYPE row_start = A_row_offsets[row];
    	INDEX_TYPE row_end = A_row_offsets[row + 1];

    	VALUE_TYPE sum = 0;
        for(INDEX_TYPE j=row_start; j < row_end; ++j)
        {
        	INDEX_TYPE col = A_column_indices[j];
        	if(x[col] != 0)
	    		sum = 1;
        }

        for(INDEX_TYPE j=1; j < coo_end; ++j)
        {
        	if(row == A_coo_row_indices[j])
        	{
        		INDEX_TYPE col = A_coo_column_indices[j];
	        	if(x[col] != 0)
		    		sum = 1;
        	}
        }

        y[row] = sum;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
__global__ void 
ell_add(	const INDEX_TYPE num_rows,
			const INDEX_TYPE num_cols,
			const INDEX_TYPE A_num_cols_per_row,
			const INDEX_TYPE B_num_cols_per_row,
			const INDEX_TYPE C_num_cols_per_row,
			const INDEX_TYPE A_pitch,
			const INDEX_TYPE B_pitch,
			const INDEX_TYPE C_pitch,
			const INDEX_TYPE *A_column_indices,
			const INDEX_TYPE *B_column_indices,
			INDEX_TYPE *C_column_indices,
			VALUE_TYPE *C_values)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	//__shared__ int entry_count[BLOCK_THREADS];
	const INDEX_TYPE invalid_index = cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory>::invalid_index;

	//entry_count[tID] = 0;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		int a_index = row;
		int b_index = row;
		int c_index = row;
		int A_col = A_column_indices[a_index];
		int B_col = B_column_indices[b_index];
		for(int n=0; n < C_num_cols_per_row; ++n, c_index+=C_pitch)
		{
			if(A_col == invalid_index && B_col == invalid_index)
			{
				C_column_indices[c_index] = invalid_index;
			}
			else if(A_col != invalid_index && B_col == invalid_index)
			{
				C_column_indices[c_index] = A_col;
				C_values[c_index] = 1;
				
				a_index += A_pitch;

				if(a_index < A_num_cols_per_row * A_pitch)
					A_col = A_column_indices[a_index];
				else
					A_col = invalid_index;
				//entry_count[tID]++;
			}
			else if(A_col == invalid_index && B_col != invalid_index)
			{
				C_column_indices[c_index] = B_col;
				C_values[c_index] = 1;
				
				b_index += B_pitch;

				if(b_index < B_num_cols_per_row * B_pitch)
					B_col = B_column_indices[b_index];
				else
					B_col = invalid_index;
				//entry_count[tID]++;
			}
			else if(A_column_indices[a_index] < B_column_indices[b_index])
			{
				C_column_indices[c_index] = A_col;
				C_values[c_index] = 1;
				
				a_index += A_pitch;

				if(a_index < A_num_cols_per_row * A_pitch)
					A_col = A_column_indices[a_index];
				else
					A_col = invalid_index;
				//entry_count[tID]++;
			}
			else if(B_column_indices[b_index] < A_column_indices[a_index])
			{
				C_column_indices[c_index] = B_col;
				C_values[c_index] = 1;
				
				b_index += B_pitch;

				if(b_index < B_num_cols_per_row * B_pitch)
					B_col = B_column_indices[b_index];
				else
					B_col = invalid_index;
				//entry_count[tID]++;
			}
			else if(A_column_indices[a_index] == B_column_indices[b_index])
			{
				C_column_indices[c_index] = A_col;
				C_values[c_index] = 1;
				
				a_index += A_pitch;
				b_index += B_pitch;

				if(a_index < A_num_cols_per_row * A_pitch)
					A_col = A_column_indices[a_index];
				else
					A_col = invalid_index;
				if(b_index < B_num_cols_per_row * B_pitch)
					B_col = B_column_indices[b_index];
				else
					B_col = invalid_index;
				//entry_count[tID]++;
			}
		}
	}
	//__syncthreads();

	// reduce_sum<INDEX_TYPE>(entry_count, BLOCK_THREADS, tID);
	// if(tID == 0)
	// 	*C_num_entries = entry_count[0];
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
<<<<<<< HEAD
__launch_bounds__(BLOCK_THREADS_MAX,1)
=======
__launch_bounds__(BLOCK_THREADS,1)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
__global__ void 
LoadEllMatrix(	const INDEX_TYPE num_rows,
				const INDEX_TYPE num_entries,
				const INDEX_TYPE num_cols_per_row,
				const INDEX_TYPE pitch,
				const INDEX_TYPE *src_row_offsets,
				const INDEX_TYPE *src_column_indices,
				const VALUE_TYPE *src_values,
				INDEX_TYPE *dst_column_indices,
				VALUE_TYPE *dst_values)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory>::invalid_index;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE row_start = src_row_offsets[row];
		INDEX_TYPE row_end = src_row_offsets[row + 1];
		INDEX_TYPE offset = row;

		dst_column_indices[row] = (row_end - row_start);
		for(int j=row_start; j < row_end; ++j, offset+=pitch)
		{
			dst_column_indices[offset] = src_column_indices[j];
			dst_values[offset] = src_values[j];
		}

		while(offset < num_cols_per_row * pitch)
		{
			dst_column_indices[offset] = invalid_index;
			offset += pitch;
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
<<<<<<< HEAD
__launch_bounds__(BLOCK_THREADS_MAX,1)
=======
__launch_bounds__(BLOCK_THREADS,1)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
__global__ void 
LoadHybMatrix(	const INDEX_TYPE num_rows,
				const INDEX_TYPE num_entries,
				const INDEX_TYPE num_cols_per_row,
				const INDEX_TYPE pitch,
				const INDEX_TYPE *src_row_offsets,
				const INDEX_TYPE *src_column_indices,
				INDEX_TYPE *dst_row_sizes,
				INDEX_TYPE *dst_column_indices,
				INDEX_TYPE *dst_coo_row_indices,
				INDEX_TYPE *dst_coo_column_indices)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory>::invalid_index;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE row_start = src_row_offsets[row];
		INDEX_TYPE row_end = src_row_offsets[row + 1];
		INDEX_TYPE offset = row;

<<<<<<< HEAD
		dst_row_sizes[row] = (row_end - row_start);
		for(int j=row_start; j < row_end; ++j, offset+=pitch)
		{
			if(offset < num_cols_per_row * pitch)
			{
				dst_column_indices[offset] = src_column_indices[j];
=======
		dst_column_indices[row] = (row_end - row_start);
		for(int j=row_start; j < row_end; ++j, offset+=pitch)
		{
			if(offset < num_cols_per_row * pitch)
			{	
				dst_column_indices[offset] = src_column_indices[j];
				atomicAdd(&dst_row_sizes[row], 1);
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
			}
			else
			{
				INDEX_TYPE index = atomicAdd(&dst_coo_column_indices[0],1)+1;
				dst_coo_column_indices[index] = src_column_indices[j];
				dst_coo_row_indices[index] = row;
			}
		}

		while(offset < num_cols_per_row * pitch)
		{
			dst_column_indices[offset] = invalid_index;
			offset += pitch;
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
<<<<<<< HEAD
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
LoadDellMatrix_B(	const INDEX_TYPE num_rows,
					const INDEX_TYPE chunk_size,
					const INDEX_TYPE chunk_length,
					const INDEX_TYPE *src_row_offsets,
					const INDEX_TYPE *src_column_indices,
					INDEX_TYPE *dst_ci,
					INDEX_TYPE *dst_cl,
					INDEX_TYPE *dst_column_indices)
=======
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
LoadDellMatrix(	const INDEX_TYPE num_rows,
				const INDEX_TYPE row_size,
				const INDEX_TYPE *src_row_offsets,
				const INDEX_TYPE *src_column_indices,
				INDEX_TYPE *dst_row_offsets,
				INDEX_TYPE *dst_column_indices,
				INDEX_TYPE *dst_row_sizes,
				INDEX_TYPE *dst_coo_row_indices,
				INDEX_TYPE *dst_coo_column_indices)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory>::invalid_index;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
<<<<<<< HEAD
		INDEX_TYPE cidx = row / chunk_size;
		INDEX_TYPE src_row_start = src_row_offsets[row];
		INDEX_TYPE src_row_end = src_row_offsets[row + 1];
		INDEX_TYPE dst_row_start = cidx + (row % chunk_size)*dst_cl[cidx];
		INDEX_TYPE dst_row_end = dst_row_start + dst_cl[cidx];

=======
		INDEX_TYPE src_row_start = src_row_offsets[row];
		INDEX_TYPE src_row_end = src_row_offsets[row + 1];
		INDEX_TYPE dst_row_start = row * row_size;
		INDEX_TYPE dst_row_end = (row+1) * row_size;

		dst_row_offsets[row+1] = dst_row_end;
		dst_row_sizes[row] = (src_row_end - src_row_start);
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
		INDEX_TYPE offset;
		for(offset=0; offset < src_row_end - src_row_start; ++offset)
		{
			if(offset < (dst_row_end - dst_row_start))
			{
				dst_column_indices[dst_row_start + offset] = src_column_indices[src_row_start + offset];
			}
			else
			{
<<<<<<< HEAD
				//overflow...   shouldn't happen on initialization
=======
				INDEX_TYPE index = atomicAdd(&dst_coo_column_indices[0],1)+1;
				dst_coo_column_indices[index] = src_column_indices[src_row_start + offset];
				dst_coo_row_indices[index] = row;
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
			}
		}

		while(offset < (dst_row_end - dst_row_start))
		{
			dst_column_indices[offset] = invalid_index;
			offset++;
		}
<<<<<<< HEAD

		dst_cl[cidx] = chunk_length;
=======
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
<<<<<<< HEAD
__launch_bounds__(BLOCK_THREADS_MAX,1)
=======
__launch_bounds__(BLOCK_THREADS,1)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
__global__ void 
CalcRowOffsets(	const INDEX_TYPE num_rows,
				const INDEX_TYPE num_cols_per_row,
				const INDEX_TYPE pitch,
				const INDEX_TYPE *src_row_offsets,
				const INDEX_TYPE *src_ELL_column_indices,
				INDEX_TYPE *dst_row_offsets)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory>::invalid_index;

	if(tID == 0)
		dst_row_offsets[0] = 0;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		dst_row_offsets[row+1] = src_row_offsets[row+1] - src_row_offsets[row] + src_ELL_column_indices[row];
<<<<<<< HEAD
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
=======
	}	
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
__global__ void 
CalcRowOffsets(	const INDEX_TYPE num_rows,
				const INDEX_TYPE *old_row_offsets,
				INDEX_TYPE *new_row_offsets,
				const INDEX_TYPE *column_indices,
				const INDEX_TYPE threshold)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = -1;

	if(tID == 0)
		new_row_offsets[0] = 0;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE row_start = old_row_offsets[row];
		INDEX_TYPE row_end = old_row_offsets[row + 1];
		INDEX_TYPE count = 0;

		for(int i=row_start; i < row_end; ++i)
		{
			if(column_indices[i] != invalid_index)
				count++;
		}

		if(count == (row_end - row_start))
			new_row_offsets[row+1] = 4*(row_end - row_start) + row_start;
		else if(count >= (row_end - row_start)/2)
			new_row_offsets[row+1] = 2*(row_end - row_start) + row_start;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
<<<<<<< HEAD
__launch_bounds__(BLOCK_THREADS_MAX,1)
=======
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
RebuildDellMatrix(	const INDEX_TYPE num_rows,
					const INDEX_TYPE *src_row_offsets,
					const INDEX_TYPE *dst_row_offsets,
					const INDEX_TYPE *src_column_indices,
					INDEX_TYPE *dst_column_indices)
{
	const int tID = threadIdx.x & (WARP_SIZE-1); 									//thread ID
	const int wID = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;			//warp ID
	const int grid_size = (blockDim.x * gridDim.x) / WARP_SIZE;

	for(int row=wID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE dst_start = dst_row_offsets[row];
		INDEX_TYPE src_start = src_row_offsets[row];
		INDEX_TYPE src_end = src_row_offsets[row+1];

		for(int j=src_start; j < src_end; j+=WARP_SIZE)
		{
			dst_column_indices[dst_start + j] = src_column_indices[src_start + j];
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
__global__ void 
RebuildMatrix(	const INDEX_TYPE num_rows,
				const INDEX_TYPE num_cols_per_row,
				const INDEX_TYPE pitch,
				const INDEX_TYPE *src_row_offsets,
				const INDEX_TYPE *src_CSR_column_indices,
				const INDEX_TYPE *src_ELL_column_indices,
				const INDEX_TYPE *dst_row_offsets,
				INDEX_TYPE *dst_column_indices)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory>::invalid_index;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE dst_index = dst_row_offsets[row];

		INDEX_TYPE src_start = src_row_offsets[row];
		INDEX_TYPE src_end = src_row_offsets[row+1];
		for(int j=src_start; j < src_end; ++j)
		{
			dst_column_indices[dst_index] = src_CSR_column_indices[j];
			dst_index++;
		}

		INDEX_TYPE ell_count = src_ELL_column_indices[row];
		INDEX_TYPE offset = row + 1;
		for(int j=0; j < ell_count; ++j)
		{
			dst_column_indices[dst_index] = src_ELL_column_indices[offset];
			dst_index++;
			offset += pitch;
		}
	}
}

<<<<<<< HEAD
}	//namespace device

=======
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
#endif
