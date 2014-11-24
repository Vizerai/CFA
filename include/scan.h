#ifndef SCAN_H
#define SCAN_H

#define NUM_BANKS 		32
#define LOG_NUM_BANKS	5
#define CONFLICT_FREE_OFFSET(n)		(n >> LOG_NUM_BANKS)

<<<<<<< HEAD
#define BITONIC_INDEX_REVERSE(i,j,gs,tID)		i = (tID/(gs>>1))*gs + (tID & ((gs>>1)-1));  j = (tID/(gs>>1))*gs + (gs - (tID & ((gs>>1)-1)) - 1)
#define BITONIC_INDEX(i,j,gs,tID)				i = (tID/(gs>>1))*gs + (tID & ((gs>>1)-1));  j = (gs>>1) + i
#define SWAP(X,Y,T)								T = X; X = Y; Y = T

template<typename T>
__device__ inline void warpScan(T &var, const int wID)
{
	T temp;

	temp = __shfl_up(var, 1);
	if(wID > 0)
		var += temp;
	temp = __shfl_up(var, 2);
	if(wID > 1)
		var += temp;
	temp = __shfl_up(var, 4);
	if(wID > 3)
		var += temp;
	temp = __shfl_up(var, 8);
	if(wID > 7)
		var += temp;
	temp = __shfl_up(var, 16);
	if(wID > 15)
		var += temp;
}

template<typename T>
__device__ inline void warpSort32(T *data, const int wID)
{
	if(wID < 16)
	{
		T var;
		int i,j;

		BITONIC_INDEX(i,j,2,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		
		BITONIC_INDEX_REVERSE(i,j,4,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,2,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }

		BITONIC_INDEX_REVERSE(i,j,8,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,4,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,2,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }

		BITONIC_INDEX_REVERSE(i,j,16,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,8,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,4,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,2,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }

		BITONIC_INDEX_REVERSE(i,j,32,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,16,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,8,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,4,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
		BITONIC_INDEX(i,j,2,wID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], var); }
	}
}

template<typename T>
__device__ inline void warpSort32_Tuple(T *key, T *data, const int wID)
{
	if(wID < 16)
	{
		T var;
		int i,j;

		BITONIC_INDEX(i,j,2,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		
		BITONIC_INDEX_REVERSE(i,j,4,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,2,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}

		BITONIC_INDEX_REVERSE(i,j,8,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,4,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,2,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}

		BITONIC_INDEX_REVERSE(i,j,16,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,8,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,4,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,2,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}

		BITONIC_INDEX_REVERSE(i,j,32,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,16,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,8,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,4,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
		BITONIC_INDEX(i,j,2,wID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], var); 
			SWAP(data[i], data[j], var);
		}
	}
}

template<typename T>
__device__ inline void warpSort64(T *data, const int wID)
{
	T var;
	int i,j;

	BITONIC_INDEX(i,j,2,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	
	BITONIC_INDEX_REVERSE(i,j,4,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,2,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }

	BITONIC_INDEX_REVERSE(i,j,8,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,4,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,2,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }

	BITONIC_INDEX_REVERSE(i,j,16,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,8,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,4,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,2,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }

	BITONIC_INDEX_REVERSE(i,j,32,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,16,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,8,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,4,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,2,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }

	BITONIC_INDEX_REVERSE(i,j,64,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,32,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,16,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,8,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,4,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
	BITONIC_INDEX(i,j,2,wID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], var); }
}

template<typename T>
__device__ inline void prescanE(T *g_odata, T *g_idata, T *temp, const int n)
=======
template<typename T>
__device__ void prescanE(T *g_odata, T *g_idata, T *temp, const int n)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
{
	int tID = threadIdx.x;  
	int offset = 1;
	
	int ai = tID;  							// load input into shared memory  
	int bi = tID + (n/2);  
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];
	 	
	for(int d = n>>1; d > 0; d >>= 1)		// build sum in place up the tree  
	{
		__syncthreads();  
		if(tID < d)
		{
			int ai = offset*(2*tID+1)-1;  
			int bi = offset*(2*tID+2)-1;  
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai]; 
		}
		offset <<= 1;
	}

	if(tID==0)
	{ temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }		// clear the last element
	 	
	for(int d = 1; d < n; d *= 2)			// traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();  

		if(tID < d)                  
		{
			int ai = offset*(2*tID+1)-1;  
			int bi = offset*(2*tID+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			T t = temp[ai];  
			temp[ai] = temp[bi];  
			temp[bi] += t;   
		}
	}
	__syncthreads();

	g_odata[ai] = temp[ai + bankOffsetA];	// write results to device memory  
	g_odata[bi] = temp[bi + bankOffsetB];
}

template<typename T>
<<<<<<< HEAD
__device__ inline void prescanE(T *data, const int n)
=======
__device__ void prescanE(T *data, const int n)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
{
	int tID = threadIdx.x;  
	int offset = 1;
	 	
	for(int d = n>>1; d > 0; d >>= 1)		// build sum in place up the tree  
	{
		__syncthreads();  
		if(tID < d)
		{
			int ai = offset*(2*tID+1)-1;  
			int bi = offset*(2*tID+2)-1;  
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			data[bi] += data[ai]; 
		}
		offset <<= 1;
	}

	if(tID==0)
	{ data[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }		// clear the last element
	 	
	for(int d = 1; d < n; d *= 2)			// traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();  

		if(tID < d)                  
		{
			int ai = offset*(2*tID+1)-1;  
			int bi = offset*(2*tID+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			T t = data[ai];  
			data[ai] = data[bi];  
			data[bi] += t;   
		}
	}
	__syncthreads();
}

template<typename T>
<<<<<<< HEAD
__device__ inline void prescanI(T *data, const int n)
=======
__device__ void prescanI(T *data, const int n)
>>>>>>> 34af523d93e062575f7e92a63da63d3f27fce1fb
{
	int tID = threadIdx.x;  
	int offset = 1;
	 	
	for(int d = n>>1; d > 0; d >>= 1)		// build sum in place up the tree  
	{
		__syncthreads();
		if(tID < d)
		{
			int ai = offset*(2*tID+1)-1;
			int bi = offset*(2*tID+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			data[bi] += data[ai]; 
		}
		offset <<= 1;
	}

	offset >>= 1;
	for(int d = 2; d < n; d *= 2)			// traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();

		if(tID < d-1)
		{
			int ai = offset*(2*tID+2)-1;  
			int bi = offset*(2*tID+3)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
  
			data[bi] += data[ai];   
		}
	}
	__syncthreads();
}

#endif