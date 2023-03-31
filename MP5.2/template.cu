// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

//changed to 1024 so we guaruntee only 1 middle block,
//this is b/c cap is 2048 x 2048, so 1st pass has max 2048 blocks
//as each thread will load 2 values. Then, for 2nd pass, since there were
//max 2048 blocks from before, there are only 2048 intermediary values 
//which would fit into a single block which can load 2048 values
#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void BKScan(float *input, float *output, int len, float *inter) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float partialSum[2*BLOCK_SIZE];

  //set up shared mem
  int i = 2*(blockIdx.x*blockDim.x + threadIdx.x);
  if (i < len){
    partialSum[2*threadIdx.x] = input[i];
  }
  else{
    partialSum[2*threadIdx.x] = 0;
  }

  if (i+1 < len){
    partialSum[2*threadIdx.x+1] = input[i+1];
  }
  else{
    partialSum[2*threadIdx.x+1] = 0;
  }
  
  //Brent-Kung Algo
  //Parallel Scan
  int stride = 1;
  int index = 0;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
      partialSum[index] += partialSum[index-stride];
    stride = stride*2;
  }

  //Post Scan
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE)
      partialSum[index+stride] += partialSum[index];
    stride = stride / 2;
  }
  //REQUIRED bc threads were writing to output before calculation was finished
  __syncthreads();
  
  //set output
  if(i<len)
    output[i] = partialSum[2*threadIdx.x];
  
  if(i+1<len)
   output[i+1] = partialSum[2*threadIdx.x+1];
  

  //set intermediates if they're not NULL
  if(threadIdx.x == 0 && inter != NULL){
    inter[blockIdx.x] = partialSum[2*BLOCK_SIZE-1];
  }
}

//Kernel to add all intermediate block values to final output
//**NOTE** Overwrites inout via +=
__global__ void pAdd(float *input, int len, float *inter){
  __shared__ float shInput[BLOCK_SIZE*2];

  //skipping 1st block of scan
  int i = 2*((blockIdx.x+1) * BLOCK_SIZE + threadIdx.x);

  //setup shared mem
  if(i<len)
    shInput[2*threadIdx.x] = input[i];
  else
    shInput[2*threadIdx.x] = 0;

  if(i+1<len)
    shInput[2*threadIdx.x+1] = input[i+1];
  else
    shInput[2*threadIdx.x+1] = 0;

  //add intermediary values to each block, it'd normally be inter[blockIdx.x-1] but we skipped 1st block, remember
  shInput[2*threadIdx.x] += inter[blockIdx.x];
  shInput[2*threadIdx.x+1] += inter[blockIdx.x];

  //overwrite value back to input array
  input[i] = shInput[2*threadIdx.x];
  input[i+1] = shInput[2*threadIdx.x+1];

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

  int num_blocks = ceil((float)numElements/(2*BLOCK_SIZE));
  //1st kernel is on all data
  dim3 DimGrid1(num_blocks,1,1);
  dim3 DimBlock1(BLOCK_SIZE,1,1);

  //2nd kernel will be on intermediary sums, of wich you have num_blocks of
  //should always dispatch 1 block
  dim3 DimGrid2(ceil((float)num_blocks/BLOCK_SIZE),1,1);
  dim3 DimBlock2(BLOCK_SIZE,1,1);

  //3rd add kernel will have a 1 to 1 correspondence to 1st kernel blocks (but don't need to add inters to 1st block)
  dim3 DimGrid3(num_blocks-1,1,1);
  dim3 DimBlock3(BLOCK_SIZE,1,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  
  //need to alloc additional global memory for intermediary sums
  cudaMalloc((void **)&deviceInter, num_blocks * sizeof(float));
  cudaMalloc((void **)&deviceInterOut, num_blocks * sizeof(float));

  //Kogge-Stone kernel scan of all input
  BKScan<<<DimGrid1, DimBlock1>>>(deviceInput, deviceOutput, numElements, deviceInter);

  //Kogge-Stone kernel scan of intermediary sums
  BKScan<<<DimGrid2, DimBlock2>>>(deviceInter, deviceInterOut, num_blocks, NULL);

  //parallel add kernel
  pAdd<<<DimGrid3, DimBlock3>>>(deviceOutput, numElements, deviceInterOut);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceInter);
  cudaFree(deviceInterOut);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
