#include <stdio.h>
#include <iostream.h>
#include <assert.h>

void checkCUDAError(const char* msg);

__global__ reverseArrayShared(int *a, int *b){

	extern __shared__ int s_data[];

	int NrPosition = blockDim.x * blockIdx.x + threadIdx.x;
	s_data[blockDim.x - 1 - threadIdx.x] = a[NrPosition];

	__syncthreads();

	int RPosition = blockDim.x * (gridDim.x - 1 - blockIdx.x) + threadIdx.x;
	b[RPosition] = s_data[threadIdx.x];

}

int main(){

	int *d_a, *d_b;

	int numThreads = 9;
	int numBlocks = 9;
	int dimA = numBlocks * numThreads;

	int h_a[dimA], h_b[dimA];

	cudaMalloc(&d_a, dimA * sizeof(int));
	cudaMalloc(&d_b, dimA * sizeof(int));

	for (int i = 0; i < dimA; ++i)
	{
		h_a[i] = i;
	}

	int sharedMemSize = 9 * sizeof(int);

	cudaMemcpy(d_a, h_a, dimA * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreads);

	reverseArrayShared <<<dimGrid, dimBlock, sharedMemSize>>> (d_a, d_b);

	checkCUDAError('Kernel invocation');

	cudaMemcpy(h_b, d_b, dimA * sizeof(int), cudaMemcpyDeviceToHost);

	checkCUDAError('Copy error');

	for (int i = 0; i < dimA; ++i)
	{
		cout << h_b[i];
	}

	cudaFree(d_a);
	cudaFree(d_b);

	return 0;
}


void checkCUDAError(const char *msg){

	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		cout << msg;
		exit(EXIT_FAILURE);
	}

}

