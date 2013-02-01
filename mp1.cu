// MP 1
#include  <wb.h>

#if (__CUDA_ARCH__ > 130)
  static const unsigned int THREADS_PER_BLOCK = 1024;
#else
  static const unsigned int THREADS_PER_BLOCK = 512;
#endif

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
  register int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) out[i] = in1[i] + in2[i];
}

int main(int argc, char ** argv) {
  wbArg_t args;
  int inputLength;
  float * hostInput1;
  float * hostInput2;
  float * hostOutput;
  float * deviceInput1;
  float * deviceInput2;
  float * deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
    size_t size = inputLength * sizeof(float);
    wbLog(TRACE, "We are gonna allocate ", size, " bytes parts");
    cudaMalloc((void **) &deviceInput1, size);
    cudaMalloc((void **) &deviceInput2, size);
    cudaMalloc((void **) &deviceOutput, size);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 gridBlocks((inputLength-1)/THREADS_PER_BLOCK + 1, 1, 1);
  dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
    vecAdd<<<gridBlocks, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaThreadSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
