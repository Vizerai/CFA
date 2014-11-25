#ifndef SPARSE_UPDATE_DEVICE
#define SPARSE_UPDATE_DEVICE

#define QUEUE_SIZE		512
#define WARP_SIZE 		32
#define LOG_WARP_SIZE	5

namespace device
{

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
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
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
OuterProductAdd_ELL_B(	const VALUE_TYPE *a,
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
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
OuterProductAdd_HYB_B(	const VALUE_TYPE *a,
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
			INDEX_TYPE offset = row;
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
					INDEX_TYPE index = atomicAdd(&row_sizes[row], 1);
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
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
OuterProductAdd_DELL_B(	const VALUE_TYPE *a,
						const VALUE_TYPE *b,
						const INDEX_TYPE *index_count,
						const INDEX_TYPE num_rows,
						const INDEX_TYPE chunks,
						const INDEX_TYPE chunk_size,
						const INDEX_TYPE groups,
						INDEX_TYPE *ci,
						INDEX_TYPE *cl,
						INDEX_TYPE *cols,
						INDEX_TYPE *overflow_col,
						INDEX_TYPE *overflow_row)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = -1;

	INDEX_TYPE num_entries_a = index_count[0];
	INDEX_TYPE num_entries_b = index_count[1];

	for(INDEX_TYPE j=tID; j < num_entries_a; j+=grid_size)
	{
		INDEX_TYPE row = a[j];
		INDEX_TYPE cID = row / chunk_size;		//chunk index
		for(INDEX_TYPE k=0; k<num_entries_b; ++k)
		{
			VALUE_TYPE b_col = b[k];
			bool overflow = false;
			bool valid = true;
			for(INDEX_TYPE c=0; c < groups && valid; c++)
			{
				INDEX_TYPE offset = ci[c*chunks + cID] + (row % chunk_size);
				for(INDEX_TYPE n=0; n < cl[c*chunks + cID] && valid; ++n)
				{
					INDEX_TYPE col = cols[offset + n];
					if(col == b_col)
					{
						valid = false;
						break;
					}
					else if(col == invalid_index)
					{
						cols[offset + n] = b_col;
						break;
					}
					else if(c == groups-1 && n == cl[c*chunks + cID]-1)
					{
						overflow = true;
					}
				}
			}

			//coordinate overflow
			if(overflow)
			{
				bool valid = true;
				for(int i=1; i < overflow_col[0]+1; ++i)
				{
					if(overflow_col[i] == b_col && overflow_row[i] == row)
					{
						valid = false;
						break;
					}
				}

				if(valid)
				{
					int index = overflow_col[0]+1;
					if(atomicCAS(&overflow_col[index], -1, b_col) == -1)
					{
						atomicAdd(&overflow_col[0], 1);
						overflow_row[index] = row;
						overflow_col[index] = b_col;
					}
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
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
OuterProductAdd_Queue(	const VALUE_TYPE *a,
						const VALUE_TYPE *b,
						const INDEX_TYPE *index_count,
						INDEX_TYPE *queue)
{
	//const int tID = threadIdx.x & (WARP_SIZE-1); 									//thread ID
	//const int wID = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;			//warp ID
	//const int grid_size = (blockDim.x * gridDim.x) / WARP_SIZE;
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	//const INDEX_TYPE invalid_index = -1;

	INDEX_TYPE num_entries_a = index_count[0];
	INDEX_TYPE num_entries_b = index_count[1];
	//INDEX_TYPE qstart = queue[0];
	INDEX_TYPE qstop = queue[1];

	if(tID == 0)
	{
		queue[qstop] = num_entries_a;
		queue[qstop+1] = num_entries_b;
	}
	__syncthreads();

	for(INDEX_TYPE j=tID; j < num_entries_a; j+=grid_size)
	{
		queue[qstop+2 + j] = a[j];
	}
	
	for(INDEX_TYPE k=tID; k<num_entries_b; k+=grid_size)
	{
		queue[qstop+2 + num_entries_a + k] = b[k];
	}
	__syncthreads();

	if(tID == 0)
	{
		queue[1] += (num_entries_a + num_entries_b + 2);
	}
	__syncthreads();
}

//Update dell matrix with entries from a CSR matrix
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
Update_DELL_B(	const INDEX_TYPE num_rows,
				const INDEX_TYPE num_chunks,
				const INDEX_TYPE chunk_size,
				const INDEX_TYPE *update_rows,
				const INDEX_TYPE *update_cols,
				const INDEX_TYPE *update_offsets,
				const INDEX_TYPE total_updates,
				INDEX_TYPE *Matrix_MD,
				INDEX_TYPE *ci,
				INDEX_TYPE *cl,
				INDEX_TYPE *ca,
				INDEX_TYPE *cols)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = -1;

	for(int idx=tID; idx < total_updates; idx++)
	{
		INDEX_TYPE count = update_offsets[idx+1] - update_offsets[idx];
		for(int n=0; n < count; n++)
		{
			INDEX_TYPE row = update_rows[idx];
			INDEX_TYPE col = update_cols[update_offsets[idx] + n];
			bool found = false;

			for(int c=0; c < num_chunks && !found; c--)
			{
				INDEX_TYPE cID = c*num_chunks + (row / chunk_size);
				INDEX_TYPE offset = ci[cID];
				for(int i=0; i < cl[cID] && !found; i++)
				{
					if(cols[offset + i] == col)
					{
						found = true;
						break;
					}
					else if(cols[offset + i] == invalid_index)
					{
						cols[offset + i] = col;
						found = true;
						break;
					}
				}
			}
		}
	}
}

//*****************************************************************************//
//update matrices with arrays of row, column and value indices
//*****************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
UpdateMatrix_dell_B(const INDEX_TYPE num_rows,
					const INDEX_TYPE chunk_size,
					const INDEX_TYPE pitch,
					const INDEX_TYPE *src_rows,
					const INDEX_TYPE *src_cols,
					const INDEX_TYPE N,
					INDEX_TYPE *Matrix_MD,
					INDEX_TYPE *ci,
					INDEX_TYPE *cl,
					INDEX_TYPE *ca,
					INDEX_TYPE *rs,
					INDEX_TYPE *cols)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	//const int lID = threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	//__shared__ volatile INDEX_TYPE rs_s[BLOCK_THREADS];

	for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
	{
		//rs_s[lID] = rs[row];
		//__syncthreads();

		for(INDEX_TYPE i=0; i < N; i++)
		{
			INDEX_TYPE cID = row / chunk_size;
			if(src_rows[i] == row)
			{
				INDEX_TYPE col = src_cols[i];
				//INDEX_TYPE rl = rs_s[lID];
				INDEX_TYPE rl = rs[row];
				INDEX_TYPE r_idx = 0, c_idx, offset;

				bool valid = true;
				bool next_chunk = false;

				do
				{
					//INDEX_TYPE offset = ca[cID] + (row % chunk_size)*cl[cID];
					INDEX_TYPE next_cID = ci[cID];
					offset = ca[cID] + (row % chunk_size);
					c_idx = 0;

					for(; c_idx < cl[cID] && r_idx < rl && valid; c_idx++, r_idx++)
					{
						//if(cols[offset + c_idx] == col)
						if(cols[offset + c_idx*pitch] == col)
						{
							valid = false;
							break;
						}
					}

					if(next_cID > 0 && r_idx < rl)
					{
						next_chunk = true;
						cID = next_cID;
					}
					else
						next_chunk = false;

				} while(next_chunk && valid);

				if(c_idx < cl[cID] && rl == r_idx && valid)
				{
					//cuPrintf("inserting col: %d  in row: %d\n", col, row);
					//cols[offset + c_idx] = col;
					cols[offset + c_idx*pitch] = col;
					rs[row] += 1;
					//rs_s[lID] += 1;
					valid = false;
				}
				else if(valid)
				{
					if(atomicCAS(&ci[cID], 0, -1) == 0)
					{
						INDEX_TYPE chunk_length = 2*cl[cID];
						INDEX_TYPE new_add = atomicAdd(&Matrix_MD[1], chunk_size*chunk_length);
						INDEX_TYPE new_cID = atomicAdd(&Matrix_MD[0], 1);

						//allocate new block...
						cl[new_cID] = chunk_length;
						ca[new_cID] = new_add;
						ci[cID] = new_cID;
					}
					
					while(ci[cID] <= 0) {}

					if(ci[cID] > 0)
					{
						cID = ci[cID];
						INDEX_TYPE offset = ca[cID] + (row % chunk_size);

						if(rl == r_idx)
						{
							//cuPrintf("inserting col: %d  in row: %d\n", col, row);
							cols[offset] = col;
							rs[row] += 1;
							//rs_s[lID] += 1;
						}
					}
				}
			}
		}

		//rs[row] = rs_s[lID];
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
UpdateMatrixW_dell_B(	const INDEX_TYPE num_rows,
						const INDEX_TYPE chunk_size,
						const INDEX_TYPE *src_rows,
						const INDEX_TYPE *src_cols,
						const INDEX_TYPE N,
						const INDEX_TYPE *Matrix_MD,
						const INDEX_TYPE *ci,
						const INDEX_TYPE *cl,
						const INDEX_TYPE *ca,
						INDEX_TYPE *rs,
						INDEX_TYPE *cols)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const int wID = threadIdx.x & (WARP_SIZE - 1);
	const int gID = threadIdx.x / WARP_SIZE;

	__shared__ volatile int warp_pool[8*WARP_SIZE*4];
	__shared__ volatile int pool_size[8];

	pool_size[gID] = 0;

	for(INDEX_TYPE n=tID; n < num_rows; n+=grid_size)
	{
		for(INDEX_TYPE i=0; i < N; i+=WARP_SIZE)
		{

		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
UpdateMatrix_hyb_B(	const INDEX_TYPE num_rows,
					const INDEX_TYPE num_cols,
					const INDEX_TYPE num_cols_per_row,
					const INDEX_TYPE pitch,
					const INDEX_TYPE *src_rows,
					const INDEX_TYPE *src_cols,
					const INDEX_TYPE N,
					INDEX_TYPE *rs,
					INDEX_TYPE *column_indices,
					INDEX_TYPE *overflow_rows,
					INDEX_TYPE *overflow_cols)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
	{
		for(INDEX_TYPE i=0; i < N; i++)
		{
			if(src_rows[i] == row)
			{
				INDEX_TYPE offset = row;
				INDEX_TYPE col = src_cols[i];
				INDEX_TYPE rl = rs[row];
				bool valid = true;

				for(INDEX_TYPE j=0; j < rl && valid; j++)
				{
					if(column_indices[offset + j*pitch] == col)
					{
						valid = false;
						break;
					}
				}

				if(rl < num_cols_per_row && valid)
				{
					column_indices[offset + rl*pitch] = col;
					rs[row] += 1;
					valid = false;
				}
				else if(valid) 	//overflow
				{
					bool ovf_valid = true;
					for(INDEX_TYPE i=1; i <= overflow_cols[0]; ++i)
					{
						if(overflow_cols[i] == col && overflow_rows[i] == row)
						{
							ovf_valid = false;
							break;
						}
					}

					if(ovf_valid)
					{
						INDEX_TYPE index = atomicAdd(&overflow_cols[0], 1) + 1;
						overflow_rows[index] = row;
						overflow_cols[index] = col;
					}
				}
			}
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS,1)
__global__ void 
InitializeMatrix_dell_B(const INDEX_TYPE num_rows,
						const INDEX_TYPE chunks,
						const INDEX_TYPE chunk_size,
						const INDEX_TYPE chunk_length,
						const INDEX_TYPE pitch,
						INDEX_TYPE *Matrix_MD,
						INDEX_TYPE *ci,
						INDEX_TYPE *cl,
						INDEX_TYPE *ca,
						INDEX_TYPE *rs,
						INDEX_TYPE *cols)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
	{
		if(row % chunk_size == 0)
		{
			INDEX_TYPE cID = row / chunk_size;
			//ca[cID] = cID*chunk_size*chunk_length;
			ca[cID] = cID*chunk_size*chunk_length;
			cl[cID] = chunk_length;
		}
	}

	if(tID == 0)
	{
		Matrix_MD[0] = chunks;
		Matrix_MD[1] = chunks*chunk_size*chunk_length;
	}
}

}	//namespace device

#endif