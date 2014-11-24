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
spmv_dell_b(    const INDEX_TYPE num_rows,
                const INDEX_TYPE * A_ci,
                const INDEX_TYPE * A_cl,
    			const INDEX_TYPE * A_cols,
                const INDEX_TYPE chunks,
                const INDEX_TYPE chunk_size,
                const INDEX_TYPE groups,
                const VALUE_TYPE * x,
                      VALUE_TYPE * y)
{
    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
        for(int c=0; c <= groups; c++)
        {
            INDEX_TYPE cidx = row / chunk_size;
            INDEX_TYPE row_start = A_ci[c*chunks + cidx] + (row % chunk_size)*A_cl[c*chunks + cidx];
        	INDEX_TYPE row_end = row_start + A_cl[c*chunks + cidx];

        	VALUE_TYPE sum = 0;
            for(INDEX_TYPE j=row_start; j < row_end; ++j)
            {
            	INDEX_TYPE col = A_cols[j];
            	if(x[col] != 0)
    	    		sum = 1;
            }

            y[row] = sum;
        }
	}
}

}	//namespace device

#endif