// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void KSScan(float *input, float *output, int len, float *inter) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float partialSum[BLOCK_SIZE];

  //set up shared mem
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < InputSize){
    partialSum[threadIdx.x] = input[i];
  }
  else{
    partialSum[threadIdx.x] = 0;
  }

  //Kogge-Stone addition into partial sum
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (threadIdx.x >= stride) // This code has a data race condition!!!
      partialSum[threadIdx.x] += partialSum[threadIdx.x-stride];
  }
  
  //set output
  output[i] = partialSum[threadIdx.x];

  //set intermediates if they're not NULL
  if(threadIdx.x = BLOCK_SIZE-1 && inter != NULL){
    inter[blockIdx.x] = partialSum[threadIdx.x];
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceInter;  //place for intermediary 
  float *deviceInterOut;  //place for intermediary out put when it is summed
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  int num_blocks = ceil((float)numInputElements/BLOCK_SIZE);
  //1st kernel is on all data
  dim3 DimGrid1(num_blocks,1,1);
  dim3 DimBlock1(BLOCK_SIZE,1,1);

  //2nd kernel will be on intermediary sums, of wich you have num_blocks of
  dim3 DimGrid2(ceil((float)num_blocks/BLOCK_SIZE),1,1);
  dim3 DimBlock2(BLOCK_SIZE,1,1);


  dim3 DimGrid3(ceil((float)numInputElements/BLOCK_SIZE),1,1);
  dim3 DimBlock3(BLOCK_SIZE,1,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  
  //need to alloc additional global memory for intermediary sums
  cudaMalloc((void **)&deviceInter, num_blocks * sizeof(float))
  cudaMalloc((void **)&deviceInterOut, num_blocks * sizeof(float))

  //Kogge-Stone kernel scan of all input
  KSScan<<DimGrid1, DimBlock1>>(deviceInput, deviceOutput, numInputElements, deviceInter);

  //Kogge-Stone kernel scan of intermediary sums
  KSScan<<DimGrid2, DimBlock2>>(deviceInter, deviceInterOut, num_blocks, NULL);

  //parallel add kernel

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
