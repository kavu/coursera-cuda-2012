#include <wb.h>

#define wbCheck(value) {                                      \
  cudaError_t _m_cudaStat = value;                            \
  if (_m_cudaStat != cudaSuccess) {                           \
    printf("Error %s at line %d in file %s\n",                \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
    return -1;                                                \
  }                                                           \
}

#define matrixMemSize(rows, columns) \
  (rows * columns) * sizeof(float)

#ifndef THREADS
# define THREADS 32
#endif

#define MAX_BLOCKS(size) \
  (size - 1)/THREADS + 1

template <unsigned int TILE_WIDTH>
__global__ void matrixMultiplyTiled(float * A, float * B, float * C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns) {
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int col = blockIdx.x * TILE_WIDTH + tx;
  unsigned int row = blockIdx.y * TILE_WIDTH + ty;
  float acc = 0;

  for (int t = 0; t < (numAColumns-1)/TILE_WIDTH + 1; ++t) {
    unsigned int ATilePitch = t * TILE_WIDTH + tx;
    unsigned int BTilePitch = t * TILE_WIDTH + ty;

    if (row < numARows && ATilePitch < numAColumns)
      ds_A[ty][tx] = A[row * numAColumns + ATilePitch];
    else
      ds_A[ty][tx] = 0;

    if (col < numBColumns && BTilePitch < numBRows)
      ds_B[ty][tx] = B[BTilePitch * numBColumns + col];
    else
      ds_B[ty][tx] = 0;

    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) acc += ds_A[ty][k] * ds_B[k][tx];
    __syncthreads();
  }

  if (row < numCRows && col < numCColumns) C[row * numCColumns + col] = acc;
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

  int deviceID = 0; // Let's get device of the default device
  cudaDeviceProp deviceProperties;

  wbCheck( cudaGetDevice(&deviceID) );
  wbCheck( cudaGetDeviceProperties(&deviceProperties, deviceID) );

  wbLog(TRACE,
        "GPU Device ", deviceID, ": \"", deviceProperties.name,
        "\" with compute capability ",
        deviceProperties.major, ".", deviceProperties.minor);


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
    if (deviceProperties.major < 2)
      matrixMultiplyTiled<16><<<grid, threads>>>(deviceA, deviceB, deviceC,
                                                 numARows, numAColumns,
                                                 numBRows, numBColumns,
                                                 numCRows, numCColumns);
    else
      matrixMultiplyTiled<32><<<grid, threads>>>(deviceA, deviceB, deviceC,
                                                 numARows, numAColumns,
                                                 numBRows, numBColumns,
                                                 numCRows, numCColumns);

    wbCheck( cudaThreadSynchronize() );
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

  wbCheck( cudaDeviceReset() );

  return 0;
}
