#include <wb.h>

#define BLOCK_SIZE 1024

#ifndef THREADS
# define THREADS 1024
#endif

#define wbCheck(stmt) do {                          \
    cudaError_t err = stmt;                         \
    if (err != cudaSuccess) {                       \
        wbLog(ERROR, "Failed to run stmt ", #stmt); \
            return -1;                              \
        }                                           \
  } while(0)


__global__ void total(float * input, float * output, unsigned int len) {
  __shared__ float sum[2*BLOCK_SIZE];
  unsigned int i = threadIdx.x;
  unsigned int j = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float localSum = (i < len) ? input[j] : 0;
    if (j + blockDim.x < len) localSum += input[j + blockDim.x];

    sum[i] = localSum;
    __syncthreads();

  for (unsigned int step = blockDim.x / 2; step >= 1; step >>= 1) {
    if (i < step) sum[i] = localSum = localSum + sum[i + step];
    __syncthreads();
  }

  if(i == 0) output[blockIdx.x] = sum[0];
}

int main(int argc, char ** argv) {
  float * hostInput; // The input 1D list
  float * hostOutput; // The output list
  float * deviceInput;
  float * deviceOutput;
  int numInputElements; // number of elements in the input list
  unsigned int numOutputElements; // number of elements in the output list

  wbArg_t args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);
    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) numOutputElements++;
    wbCheck( cudaHostAlloc(&hostOutput, numOutputElements * sizeof(float), cudaHostAllocDefault) );
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck( cudaMalloc(&deviceInput,  numInputElements *  sizeof(float)) );
    wbCheck( cudaMalloc(&deviceOutput, numOutputElements * sizeof(float)) );
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck( cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice) );
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 grid(numOutputElements);
  dim3 threads(THREADS);

  wbTime_start(Compute, "Performing CUDA computation");
    total<<<grid, threads>>>(deviceInput, deviceOutput, numInputElements);
    wbCheck( cudaThreadSynchronize() );
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck( cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost) );
  wbTime_stop(Copy, "Copying output memory to the CPU");

  #pragma unroll
  for (unsigned int i = 1; i < numOutputElements; i++) hostOutput[0] += hostOutput[i];

  wbTime_start(GPU, "Freeing GPU Memory");
    wbCheck( cudaFree(deviceInput) );
    wbCheck( cudaFree(deviceOutput) );
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  wbCheck( cudaFreeHost(hostOutput) );

  wbCheck( cudaDeviceReset() );

  return 0;
}
