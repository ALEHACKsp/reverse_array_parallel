#include <stdio.h>
#include <assert.h>

__global__ reverseArray(int *array, int *arraytr){
    int inOffset  = blockDim.x * blockIdx.x;
    int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
    int in  = inOffset + threadIdx.x;
    int out = outOffset + (blockDim.x - 1 - threadIdx.x);
    array[out] = arraytr[in];
}


int main(int argc, char** argv){	

	int dimA = 256 * 1024; 
	int h_a[],h_b[];
	int *d_a, *d_b;

	int numThreadsPerBlock = 256;
	int numBlocks = 1024;

	cudaMalloc(&d_a, dimA * sizeof(int));
	cudaMalloc(&d_b, dimA * sizeof(int));

	for (int i = 0; i < dimA; ++i)
	{
		h_a[i] = i;
	}

	cudaMemcpy(d_a, h_a, dimA * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsPerBlock);

	reverseArray <<<dimGrid, dimBlock>>>(d_a, d_b);
	cudaThreadSynchronize();

	CudaCheckError('Kernel Invocation');

	cudaMemcpy(h_b, d_b, dimA * sizeof(int), cudaMemcpyDeviceToHost);

	CudaCheckError('Copy data error');

	for (int i = 0; i < 256*1024; ++i)
	{
		printf("%d\n", h_b[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);

	CudaCheckError('Could not free memory');

	return 0;
}


void CudaCheckError(const char *msg){

	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }     

}
