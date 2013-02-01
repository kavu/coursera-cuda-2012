#include <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#ifndef BLOCK_SIZE
# define BLOCK_SIZE 256
#endif

#define HALF_BLOCK_SIZE BLOCK_SIZE << 1

#define MEM_SIZE(size)  \
  size * sizeof(float)

__global__ void post_scan(float* in, float* add, int len) {
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

  if (blockIdx.x) {
     if (start + t < len) in[start + t] += add[blockIdx.x - 1];
     if (start + BLOCK_SIZE + t < len) in[start + BLOCK_SIZE + t] += add[blockIdx.x - 1];
  }
}

__global__ void scan(float* in, float* out, float* post, int len) {
  __shared__ float scan_array[HALF_BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
  int index;

  if (start + t < len) scan_array[t] = in[start + t];
  else scan_array[t] = 0;

  if (start + BLOCK_SIZE + t < len) scan_array[BLOCK_SIZE + t] = in[start + BLOCK_SIZE + t];
  else scan_array[BLOCK_SIZE + t] = 0;
  __syncthreads();

  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
     index = (t + 1) * stride * 2 - 1;
     if (index < 2 * BLOCK_SIZE) scan_array[index] += scan_array[index - stride];
     __syncthreads();
  }

  for (unsigned int stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
     index = (t + 1) * stride * 2 - 1;
     if (index + stride < 2 * BLOCK_SIZE) scan_array[index + stride] += scan_array[index];
     __syncthreads();
  }

  if (start + t < len) out[start + t] = scan_array[t];
  if (start + BLOCK_SIZE + t < len) out[start + BLOCK_SIZE + t] = scan_array[BLOCK_SIZE + t];

  if (post && t == 0) post[blockIdx.x] = scan_array[2 * BLOCK_SIZE - 1];
}

int main(int argc, char ** argv) {
  float* hostInput;
  float* hostOutput;
  float* deviceInput;
  float* deviceOutput;
  float* deviceScanFirstPass;
  float* deviceScanSecondPass;
  int numElements, numBlocks;
  size_t numElementsMemSize;

  int halfBlockSize = BLOCK_SIZE << 1;
  wbArg_t args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);

    numElementsMemSize = MEM_SIZE(numElements);
    numBlocks = ceil((float)numElements/halfBlockSize);

    wbCheck(cudaHostAlloc(&hostOutput, numElementsMemSize, cudaHostAllocDefault));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numElements);
  wbLog(TRACE, "The number of blocks is ", numBlocks);

  wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc(&deviceInput,          numElementsMemSize));
    wbCheck(cudaMalloc(&deviceOutput,       numElementsMemSize));
    wbCheck(cudaMalloc(&deviceScanFirstPass,  MEM_SIZE(halfBlockSize)));
    wbCheck(cudaMalloc(&deviceScanSecondPass, MEM_SIZE(halfBlockSize)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElementsMemSize));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElementsMemSize, cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 grid(numBlocks);
  dim3 threads(BLOCK_SIZE);

  wbTime_start(Compute, "Performing CUDA computation");
    scan<<<grid, threads>>>(deviceInput, deviceOutput, deviceScanFirstPass, numElements);
    cudaDeviceSynchronize();

    scan<<<1, threads>>>(deviceScanFirstPass, deviceScanSecondPass, NULL, halfBlockSize);
    cudaDeviceSynchronize();

    post_scan<<<grid, threads>>>(deviceOutput, deviceScanSecondPass, numElements);
    cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElementsMemSize, cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
    wbCheck(cudaFree(deviceInput));
    wbCheck(cudaFree(deviceOutput));
    wbCheck(cudaFree(deviceScanFirstPass));
    wbCheck(cudaFree(deviceScanSecondPass));
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  wbCheck(cudaFreeHost(hostOutput));

  return 0;
}
