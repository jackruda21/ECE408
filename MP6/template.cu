// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

//@@ insert code here
__global__ void histEq1(float *input, float *output, float *histo, int height, int width){
  //initialize shared data
  __shared__ int blockHistogram[HISTOGRAM_LENGTH];
  __shared__ unsigned char blockGrayImg[BLOCK_SIZE*BLOCK_SIZE];

  //Set indices
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = iy * width + ix;
  int block_idx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

  // Set shared block w/ grayscale values
  unsigned char r = (unsigned char) 255 * input[3*idx];
  unsigned char g = (unsigned char) 255 * input[3*idx + 1];
  unsigned char b = (unsigned char) 255 * input[3*idx + 2];
  blockGrayImg[block_idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b)

  //init histogram to 0
  blockHistogram[block_idx] = 0;

  //create local shared histogram based on image
  atomicAdd( &(blockHistogram[blockGrayImg[block_idx]]), 1);
  __syncthreads();

  //add to global histogram
  atomicAdd( &histo[block_idx] , blockHistogram[block_idx]);
}

//kernel function to compute CDF, DOUBLE CHECK THIS!!!
__global__ void computeCDF(float *hist, float *CDF, int len){

  __shared__ float partialSum[HISTOGRAM_LENGTH];
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
  while(stride < HISTOGRAM_LENGTH) {
    __syncthreads();
    index = (threadIdx.x+1)*stride*2 - 1;
    if(index < HISTOGRAM_LENGTH && (index-stride) >= 0)
      partialSum[index] += partialSum[index-stride];
    stride = stride*2;
  }

  //Post Scan
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < HISTOGRAM_LENGTH)
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
    inter[blockIdx.x] = partialSum[HISTOGRAM_LENGTH-1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInput;
  float *deviceOutput;
  float *deviceHisto;
  float *deviceCDF;


  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here

  //alloc device mem
  cudaMalloc((void**)&deviceInput, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void**)&deviceOutput, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void**)&deviceHisto, HISTOGRAM_LENGTH*sizeof(float));
  cudaMalloc((void**)&deviceCDF, HISTOGRAM_LENGTH*sizeof(float));

  //copy input image to device
  cudaMemcpy(deviceInput, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);

  //set up grid dimensions
  dim3 DimGrid1(ceil((float)imageWidth/BLOCK_SIZE),ceil((float)imageHeight/BLOCK_SIZE),1);
  dim3 DimBlock1(BLOCK_WIDTH,BLOCK_WIDTH,1);

  dim3 DimGrid2(1,1,1);
  dim3 DimBlock2(HISTOGRAM_LENGTH/2,HISTOGRAM_LENGTH/2,1);

  //call kernel
  histEq1<<<DimGrid1, DimBlock1>>>();
  computeCDF<<<DimGrid2, DimBlock2>>>(deviceHisto, deviceCDF, HISTOGRAM_LENGTH);

  //retrieve output from device
  cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice);

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
