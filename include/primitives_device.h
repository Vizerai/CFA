#ifndef PRIMITIVES_DEVICE_H
#define PRIMITIVES_DEVICE_H

#define WARP_SIZE	32

//shared memory size = BLOCK_SIZE
//shared memory reduce
template <typename VALUE_TYPE, int BLOCK_SIZE>
inline __device__ void reduce_sum(VALUE_TYPE *a, const int tID)
{
	int n = BLOCK_SIZE;
	while(n > 1)
	{
		int half = n / 2;
		if(tID < half)
			a[tID] += a[n - tID - 1];
		n = n - half;
		__syncthreads();
	}
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
FILL(	VALUE_TYPE *a,
		const VALUE_TYPE value,
		const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(int i=tID; i<size; i+=grid_size)
		a[i] = value;
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
AND_OP(	const VALUE_TYPE *A,
		const VALUE_TYPE *B,
		VALUE_TYPE *C,
		const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(int i=tID; i<size; i+=grid_size)
		C[i] = A[i] & B[i];
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
get_indices(	const VALUE_TYPE *a,
				VALUE_TYPE *b,
				const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	__shared__ int index;
	index = 0;
	__syncthreads();

	for(int i=tID; i<size; i+=grid_size)
	{
		if(a[i] != 0)
		{
			int old_index = atomicAdd(&index, 1);
			b[old_index] = i;
		}
	}
}

//b size = a size + 1
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
gather_reduce(	const VALUE_TYPE *a,
				VALUE_TYPE *b,
				INDEX_TYPE *index_count,
				const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	__shared__ int count[BLOCK_THREADS];
	__shared__ int index;
	count[tID] = 0;
	index = 0;
	__syncthreads();

	for(int i=tID; i<size; i+=grid_size)
	{
		if(a[i] != 0)
		{
			int old_index = atomicAdd(&index, 1);
			b[old_index] = i;
			count[tID]++;
		}
	}
	__syncthreads();

	reduce_sum<VALUE_TYPE, BLOCK_THREADS>(count, tID);

	if(tID == 0)
		index_count[0] = count[0];
}

template <typename INDEX_TYPE, typename VALUE_TYPE, int BLOCK_SIZE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
count(	const VALUE_TYPE *a,
		const VALUE_TYPE val,
		INDEX_TYPE *res,
		const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	__shared__ INDEX_TYPE count[BLOCK_SIZE];
	count[tID] = 0;
	__syncthreads();

	for(int i=tID; i < size; i+=grid_size)
	{
		if(a[i] == val)
			count[tID]++;
	}
	__syncthreads();

	reduce_sum<VALUE_TYPE, BLOCK_SIZE>(count, tID);

	if(tID == 0)
		res[0] = count[0];
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
AccumVec(	VALUE_TYPE *a,
			const VALUE_TYPE *b,
			const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	//a += b
	for(int i=tID; i<size; i+=grid_size)
	{
		if(b[i])
			a[i] = 1;
		//a[i] += b[i];
	}
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
InnerProductStore(	const VALUE_TYPE *a,
					const VALUE_TYPE *b,
					const int size,
					VALUE_TYPE *c)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(int index=tID; index < size; index+=grid_size)
	{
		if(a[index] != 0 && b[index] != 0)
		{
			c[0] = 1;
		}
	}
}

//a (.) Mat -> b
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void InnerProduct(	const VALUE_TYPE *a,
								VALUE_TYPE *b,
								const INDEX_TYPE size_a,
								const INDEX_TYPE size_b,
								const INDEX_TYPE num_rows,
								const INDEX_TYPE num_cols,
								const INDEX_TYPE num_cols_per_row,
								const INDEX_TYPE pitch,
								const INDEX_TYPE *column_indices)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	
	const INDEX_TYPE invalid_index = cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory>::invalid_index;

	for(int row=tID; row<size_a; row+=grid_size)
	{
		int offset = row;
		for(int n=0; n < num_cols_per_row; ++n, offset+=pitch)
		{
			int col = column_indices[offset];
			if(col != invalid_index)
			{
				if(a[row] == 1)
					b[col] = 1;
			}
		}
	}
}

#endif