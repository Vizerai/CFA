#ifndef SPMV_DEVICE
#define SPMV_DEVICE

#define QUEUE_SIZE		512

namespace device
{

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
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
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_hybb(	const INDEX_TYPE num_rows,
            const INDEX_TYPE num_cols_per_row,
            const INDEX_TYPE pitch,
            const INDEX_TYPE * A_ell_column_indices,
        	const INDEX_TYPE * A_coo_row_indices,
            const INDEX_TYPE * A_coo_column_indices,
            const INDEX_TYPE * A_rs,
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
        INDEX_TYPE rl = A_rs[row];
        INDEX_TYPE r_idx = 0;

        for(INDEX_TYPE n = 0; n < num_cols_per_row && r_idx < rl; ++n, ++r_idx)
        {
            const INDEX_TYPE col = A_ell_column_indices[offset];
            if(col != invalid_index)
            {
	            if(x[col] != 0)
	            {
                	sum = 1;
                    break;
                }
            }
            else
            	break;

            offset += pitch;
        }

        int overflow_size = A_coo_column_indices[0];
        for(int n=1; n <= overflow_size && r_idx < rl; n++)
        {
        	if(A_coo_row_indices[n] == row)
        	{
                r_idx++;
        		const INDEX_TYPE col = A_coo_column_indices[n];
        		if(x[col] != 0)
	            {
                	sum = 1;
                    break;
                }
        	}
        }

        y[row] = sum;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_hyb(   const INDEX_TYPE num_rows,
            const INDEX_TYPE num_cols_per_row,
            const INDEX_TYPE pitch,
            const INDEX_TYPE * A_ell_column_indices,
            const VALUE_TYPE * A_ell_values,
            const INDEX_TYPE * A_coo_row_indices,
            const INDEX_TYPE * A_coo_column_indices,
            const VALUE_TYPE * A_coo_values,
            const INDEX_TYPE * A_rs,
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
        INDEX_TYPE rl = A_rs[row];
        INDEX_TYPE r_idx = 0;

        for(INDEX_TYPE n = 0; n < num_cols_per_row && r_idx < rl; ++n, ++r_idx)
        {
            const INDEX_TYPE col = A_ell_column_indices[offset];
            if(col != invalid_index)
            {
                sum += A_ell_values[offset] * x[col];
            }
            else
                break;

            offset += pitch;
        }

        int overflow_size = A_coo_column_indices[0];
        for(int n=1; n <= overflow_size && r_idx < rl; n++)
        {
            if(A_coo_row_indices[n] == row)
            {
                r_idx++;
                sum += A_coo_values[n] * x[A_coo_column_indices[n]];
            }
        }

        y[row] = sum;
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
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
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_csr(   const INDEX_TYPE num_rows,
            const INDEX_TYPE * A_row_offsets,
            const INDEX_TYPE * A_column_indices,
            const VALUE_TYPE * A_values,
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
            sum += A_values[j] * x[A_column_indices[j]];
        }

        y[row] = sum;
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_dell_b(    const INDEX_TYPE num_rows,
                const INDEX_TYPE chunk_size,
                const INDEX_TYPE pitch,
                const INDEX_TYPE * Matrix_MD,
                const INDEX_TYPE * A_ci,
                const INDEX_TYPE * A_cl,
                const INDEX_TYPE * A_ca,
                const INDEX_TYPE * A_rs,
                const INDEX_TYPE * A_cols,  
                const VALUE_TYPE * x,
                      VALUE_TYPE * y)
{
    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
        INDEX_TYPE rl = A_rs[row];
        VALUE_TYPE sum = 0;
        INDEX_TYPE r_idx = 0;
        bool next_chunk = false;

        do
        {
            INDEX_TYPE cID = row / chunk_size;
            INDEX_TYPE next_cID = A_ci[cID];
            INDEX_TYPE offset = A_ca[cID] + (row % chunk_size);

            for(INDEX_TYPE c_idx = 0; c_idx < A_cl[cID]; ++c_idx, ++r_idx)
            {
                INDEX_TYPE col = A_cols[offset + c_idx*pitch];
                if(x[col] != 0)
    	    	{
                	sum = 1;
                    break;          //break out because it is a binary matrix and the value of this dot product is 1
                }
            }

            if(next_cID > 0 && r_idx < rl && sum == 0)
            {
                next_chunk = true;
                cID = next_cID;
            }
            else
                next_chunk = false;

        } while(next_chunk);
        y[row] = sum;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_dell(      const INDEX_TYPE num_rows,
                const INDEX_TYPE chunk_size,
                const INDEX_TYPE pitch,
                const INDEX_TYPE * Matrix_MD,
                const INDEX_TYPE * A_ci,
                const INDEX_TYPE * A_cl,
                const INDEX_TYPE * A_ca,
                const INDEX_TYPE * A_rs,
                const INDEX_TYPE * A_cols,
                const VALUE_TYPE * A_vals,    
                const VALUE_TYPE * x,
                      VALUE_TYPE * y)
{
    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
        VALUE_TYPE sum = 0;
        INDEX_TYPE rl = A_rs[row];
        INDEX_TYPE r_idx = 0;
        INDEX_TYPE cID = row / chunk_size;
        bool next_chunk = false;

        do
        {
            INDEX_TYPE next_cID = A_ci[cID];
            //INDEX_TYPE offset = A_ca[cID] + (row % chunk_size);
            INDEX_TYPE offset = A_ca[cID] + (row & (chunk_size-1))*pitch;
            INDEX_TYPE cl = A_cl[cID];

            for(INDEX_TYPE c_idx = 0; c_idx < cl && r_idx < rl; ++c_idx, ++r_idx)
            {
                //sum += A_vals[offset + c_idx*pitch] * x[A_cols[offset + c_idx*pitch]];
                sum += A_vals[offset + c_idx] * x[A_cols[offset + c_idx]];
            }

            if(next_cID > 0 && r_idx < rl)
            {
                next_chunk = true;
                cID = next_cID;
            }
            else
                next_chunk = false;

        } while(next_chunk);

        y[row] = sum;
    }
}

}	//namespace device

#endif