#include <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define MASK 5
#define TILE_WIDTH 16
#define RADIUS MASK/2
#define AREA (TILE_WIDTH + MASK - 1)
#define GRID_SIZE(x) (ceil((float)x/TILE_WIDTH))

__device__ inline void setIndexes(unsigned int d,
                                  unsigned int &dX,
                                  unsigned int &dY,
                                  int &sX, int &sY){
  dX = d % AREA;
  dY = d / AREA;
  sX = blockIdx.x * TILE_WIDTH + dX - RADIUS;
  sY = blockIdx.y * TILE_WIDTH + dY - RADIUS;
}

__global__ void convolution(float* I, const float* __restrict__ M, float* P,
                            int channels, int width, int height) {
  __shared__ float tmp[AREA][AREA];

  float acc;
  int sourceY, sourceX;
  unsigned int source, destination, destinationY, destinationX;
  unsigned int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
  unsigned int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

  for (unsigned int k = 0; k < channels; k++) {
    destination = threadIdx.y * TILE_WIDTH + threadIdx.x;
    setIndexes(destination,
               destinationX,
               destinationY,
               sourceX, sourceY);
    source = (sourceY * width + sourceX) * channels + k;
    if (sourceY >= 0 && sourceY < height && sourceX >= 0 && sourceX < width)
      tmp[destinationY][destinationX] = I[source];
    else
      tmp[destinationY][destinationX] = 0;

    destination = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    setIndexes(destination,
               destinationX,
               destinationY,
               sourceX, sourceY);
    source = (sourceY * width + sourceX) * channels + k;

    if (destinationY < AREA)
      if (sourceY >= 0 && sourceY < height && sourceX >= 0 && sourceX < width)
        tmp[destinationY][destinationX] = I[source];
      else
        tmp[destinationY][destinationX] = 0;

    __syncthreads();

    acc = 0;
    #pragma unroll
    for (unsigned int i = 0; i < MASK; i++)
      #pragma unroll
      for (unsigned int j = 0; j < MASK; j++)
        acc += tmp[threadIdx.y + i][threadIdx.x + j] * M[i * MASK + j];

    if (y < height && x < width) P[(y * width + x) * channels + k] = min(max(acc, 0.0), 1.0);

    __syncthreads();
  }
}

int main(int argc, char* argv[]) {
  wbImage_t inputImage, outputImage;

  int   maskRows, maskColumns, imageChannels,
        imageWidth, imageHeight;
  char  *inputImageFile, *inputMaskFile;
  float *hostInputImageData, * hostOutputImageData,
        *hostMaskData, *deviceInputImageData,
        *deviceOutputImageData, *deviceMaskData;

  wbArg_t arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile = wbArg_getInputFile(arg, 1);

  inputImage = wbImport(inputImageFile);
  hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */
  size_t maskMemSize = maskRows * maskColumns * sizeof(float);

  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  size_t imageMemSize = imageWidth * imageHeight * imageChannels * sizeof(float);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
        wbCheck( cudaMalloc((void **) &deviceInputImageData, imageMemSize) );
        wbCheck( cudaMalloc((void **) &deviceOutputImageData, imageMemSize) );
        wbCheck( cudaMalloc((void **) &deviceMaskData, maskMemSize) );
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
      wbCheck( cudaMemcpy(deviceInputImageData, hostInputImageData, imageMemSize, cudaMemcpyHostToDevice) );
      wbCheck( cudaMemcpy(deviceMaskData, hostMaskData, maskMemSize, cudaMemcpyHostToDevice) );
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
      unsigned int dimGridX = GRID_SIZE(imageWidth);
      unsigned int dimGridY = GRID_SIZE(imageHeight);
      dim3 dimGrid(dimGridX, dimGridY);
      dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
      convolution<<<dimGrid, dimBlock>>>(deviceInputImageData,
                                         deviceMaskData,
                                         deviceOutputImageData,
                                         imageChannels,
                                         imageWidth,
                                         imageHeight);
      wbCheck( cudaThreadSynchronize() );
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
      wbCheck( cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageMemSize, cudaMemcpyDeviceToHost) );
    wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  wbCheck( cudaFree(deviceInputImageData) );
  wbCheck( cudaFree(deviceOutputImageData) );
  wbCheck( cudaFree(deviceMaskData) );

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
