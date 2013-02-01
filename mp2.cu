#include <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define matrixMemSize(rows, columns) \
  (rows * columns) * sizeof(float)

#ifndef THREADS
#  define THREADS 32
#endif

#define MAX_BLOCKS(size) \
  (size-1)/THREADS + 1

__global__ void matrixMultiply(float * A, float * B, float * C,
                               int numARows, int numAColumns,
                               int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < numCRows) && (col < numCColumns)) {
    float value = 0;
#pragma unroll
    for (int k = 0; k < numAColumns; ++k)
      value += A[row * numAColumns + k] * B[k * numBColumns + col];
    C[row * numCColumns + col] = value;
  }
}

int main(int argc, char ** argv) {
  wbArg_t args;
  float * hostA; // The A matrix
  float * hostB; // The B matrix
  float * hostC; // The output C matrix
  float * deviceA;
  float * deviceB;
  float * deviceC;
  int numARows; // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows; // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows; // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

    numCRows = numARows;
    numCColumns = numBColumns;

    wbCheck( cudaMallocHost(&hostC, matrixMemSize(numCRows, numCColumns)) );
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck( cudaMalloc(&deviceA, matrixMemSize(numARows, numAColumns)) );
    wbCheck( cudaMalloc(&deviceB, matrixMemSize(numBRows, numBColumns)) );
    wbCheck( cudaMalloc(&deviceC, matrixMemSize(numCRows, numCColumns)) );
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck( cudaMemcpy(deviceA, hostA, matrixMemSize(numARows, numAColumns), cudaMemcpyHostToDevice) );
    wbCheck( cudaMemcpy(deviceB, hostB, matrixMemSize(numBRows, numBColumns), cudaMemcpyHostToDevice) );
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 grid(MAX_BLOCKS(numCColumns), MAX_BLOCKS(numCRows));
  dim3 threads(THREADS, THREADS);

  wbTime_start(Compute, "Performing CUDA computation");
    matrixMultiply<<<grid, threads>>>(deviceA, deviceB, deviceC,
                                      numARows, numAColumns,
                                      numBRows, numBColumns,
                                      numCRows, numCColumns);
    cudaThreadSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck( cudaMemcpy(hostC, deviceC, matrixMemSize(numCRows, numCColumns), cudaMemcpyDeviceToHost) );
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
    wbCheck( cudaFree(deviceA) );
    wbCheck( cudaFree(deviceB) );
    wbCheck( cudaFree(deviceC) );
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  cudaFreeHost(hostA);
  cudaFreeHost(hostB);
  cudaFreeHost(hostC);

  return 0;
}

