#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define BLOCK_WIDTH 4
//@@ Define constant memory for device kernel here
__constant__ float KFILTER[27];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  //Shared mem for 6x6x6 block
  __shared__ float input_tile[(BLOCK_WIDTH+2)*(BLOCK_WIDTH+2)*(BLOCK_WIDTH+2)];

  //define block/thread indices
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x; 
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  //Calculate x,y,z idxs of input
  int x = bx * (BLOCK_WIDTH) + tx - 1;
  int y = by * (BLOCK_WIDTH) + ty - 1;
  int z = bz * (BLOCK_WIDTH) + tz - 1;

  //Load 6x6x6 to shared mem, 0 if out of input bounds
  if(x<0 || x>=x_size || y<0 || y>=y_size || z<0 || z>=z_size){
    input_tile[tz*((BLOCK_WIDTH+2)*(BLOCK_WIDTH+2))+ty*(BLOCK_WIDTH+2)+tx] = 0;
  }
  else{
    input_tile[tz*((BLOCK_WIDTH+2)*(BLOCK_WIDTH+2))+ty*(BLOCK_WIDTH+2)+tx] = input[z * (y_size * x_size) + y * x_size + x];
  }
  __syncthreads();

  //set output value based on filter
  float outVal = 0;
  if(tx > 0 && tx < (BLOCK_WIDTH+1) && ty > 0 && ty < (BLOCK_WIDTH+1) && tz > 0 && tz < (BLOCK_WIDTH+1)){
    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
        for(int k=0; k<3; k++){
          outVal += input_tile[(tz-1+i)*((BLOCK_WIDTH+2)*(BLOCK_WIDTH+2)) + (ty-1+j)*(BLOCK_WIDTH+2) + tx-1+k] * KFILTER[i*9 + j * 3 + k];
        }
      }
    }
    if(x>=0 && x<x_size && y>=0 && y<y_size && z>=0 && z<z_size){
      output[z * (y_size * x_size) + y * x_size + x] = outVal;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions

  cudaMalloc((void**) &deviceInput, z_size*y_size*x_size * sizeof(float));
  cudaMalloc((void**) &deviceOutput, z_size*y_size*x_size * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu

  //Setting constant mem
  float filter[27];
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      for(int k=0; k<3; k++){
        filter[i*9 + j * 3 + k] = hostKernel[i * (3 * 3) + j * 3 + k];
      }
    }
  }

  cudaMemcpyToSymbol(KFILTER, filter, 27*sizeof(float));

  cudaMemcpy(deviceInput, hostInput+3, z_size*y_size*x_size * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  
  //This DimGrid implies edge threads will just do no calculations
  dim3 DimGrid(ceil((float)x_size/BLOCK_WIDTH),ceil((float)y_size/BLOCK_WIDTH),ceil((float)z_size/BLOCK_WIDTH));
  dim3 DimBlock(BLOCK_WIDTH+2,BLOCK_WIDTH+2,BLOCK_WIDTH+2);
  
  //@@ Launch the GPU kernel here

  conv3d<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");
  
  cudaMemcpy(hostOutput+3, deviceOutput, z_size*y_size*x_size * sizeof(float), cudaMemcpyDeviceToHost);
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;

  // if(z_size == 31 && y_size==17){
  //   for(int i=0; i<z_size; i++){
  //     for(int j=0; j<y_size; j++){
  //       for(int k=0; k<x_size; k++){
  //         wbLog(TRACE, hostOutput[i*8*8 + j*8 + k]);
  //       }
  //     }
  //   }
  //   wbLog(TRACE, '\n');
  // }



  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
