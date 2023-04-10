// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

//@@ insert code here
__global__ void histEq1(float *input, float *output, float *histo, float* cdf, int height, int width, int* sync_blocks, int len){
  //initialize shared data
  __shared__ int blockHistogram[HISTOGRAM_LENGTH];
  __shared__ unsigned char blockGrayImg[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ unsigned char ucharInputRed[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ unsigned char ucharInputGreen[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ unsigned char ucharInputBlue[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float partialCdf[HISTOGRAM_LENGTH];

  //Set indices
  int ix = blockIdx.x * blockDim.x + threadIdx.x; //x ccoord in input image
  int iy = blockIdx.y * blockDim.y + threadIdx.y; //y coord in input image
  int idx = iy * width + ix;  //index into input image (only x,y, not channel)
  int block_idx = threadIdx.y * BLOCK_SIZE + threadIdx.x; //index from 0-255 within block
  int error_timer;

  ucharInputRed[block_idx] = (unsigned char) 255 * input[3*idx];
  ucharInputGreen[block_idx] = (unsigned char) 255 * input[3*idx + 1];
  ucharInputBlue[block_idx] = (unsigned char) 255 * input[3*idx + 2];

  // Set shared block w/ grayscale values
  blockGrayImg[block_idx] = (unsigned char) (0.21*ucharInputRed[block_idx] + 0.71*ucharInputGreen[block_idx] + 0.07*ucharInputBlue[block_idx]);

  //init histogram to 0
  blockHistogram[block_idx] = 0;

  //create local shared histogram based on image
  atomicAdd( &(blockHistogram[blockGrayImg[block_idx]]), 1);
  __syncthreads();

  //add to global histogram
  atomicAdd( &histo[block_idx] , blockHistogram[block_idx]);

  //init sync_blocks to 0
  *sync_blocks = 0;
  //synchronize all blocks adding to global histogram so we don't have to launch multiple kernels
  if(threadIdx.x == 0 && threadIdx.y == 0){
    atomicAdd(sync_blocks, 1);
  }
  while(*sync_blocks < gridDim.x * gridDim.y){
    error_timer++;
    if(error_timer > 10000){
      return;
    }
  }
  //reset sync_blocks to be used as a flag
  *sync_blocks = 0;

  //Parallel Scan Algorithm, only do in 1 block to create the cdf
  if(blockIdx.x == 0 && blockIdx.y == 0 && block_idx < HISTOGRAM_LENGTH / 2){
    //set up shared mem
    int i = 2*(block_idx);
    
    if (i < len){
      partialCdf[2*threadIdx.x] = input[i];
    }
    else{
      partialCdf[2*threadIdx.x] = 0;
    }

    if (i+1 < len){
      partialCdf[2*threadIdx.x+1] = input[i+1];
    }
    else{
      partialCdf[2*threadIdx.x+1] = 0;
    }
    
    //Brent-Kung Algo
    //Parallel Scan
    int stride = 1;
    int index = 0;
    while(stride < HISTOGRAM_LENGTH) {
      __syncthreads();
      index = (threadIdx.x+1)*stride*2 - 1;
      if(index < HISTOGRAM_LENGTH && (index-stride) >= 0)
        partialCdf[index] += partialCdf[index-stride];
      stride = stride*2;
    }

    //Post Scan
    stride = HISTOGRAM_LENGTH/4;
    while(stride > 0) {
      __syncthreads();
      index = (threadIdx.x+1)*stride*2 - 1;
      if ((index+stride) < HISTOGRAM_LENGTH)
        partialCdf[index+stride] += partialCdf[index];
      stride = stride / 2;
    }
    //REQUIRED bc threads were writing to output before calculation was finished
    __syncthreads();
    
    //set output cdf, p(x) built in here w/ the /(width * height)
    if(i<len)
      cdf[i] = partialCdf[2*threadIdx.x] / (width * height) ;
    
    if(i+1<len)
      cdf[i+1] = partialCdf[2*threadIdx.x+1] / (width * height);

    __syncthreads();
    //set sync_blocks flag to notify other blocks that CDF is computed and set in global mem
    *sync_blocks = 1;
  }

  //Have inactive blocks wait until 1st block is done with parallel scan
  while(*sync_blocks != 1){
    error_timer++;
    if(error_timer > 10000){
      return;
    }
  }

  //color correcting process
  ucharInputRed[block_idx] = min(max((255*(cdf[ucharInputRed[block_idx]] - cdf[0])/(1.0 - cdf[0])), 0.0), 255.0); //correct_color(ucharImage[ii]);
  ucharInputGreen[block_idx] = min(max((255*(cdf[ucharInputGreen[block_idx]] - cdf[0])/(1.0 - cdf[0])), 0.0), 255.0);
  ucharInputBlue[block_idx] = min(max((255*(cdf[ucharInputBlue[block_idx]] - cdf[0])/(1.0 - cdf[0])), 0.0), 255.0);

  if(3*idx < height * width * 3){
    output[3*idx] = (float)(ucharInputRed[block_idx]/255.0);
    output[3*idx + 1] = (float)(ucharInputGreen[block_idx]/255.0);
    output[3*idx + 2] = (float)(ucharInputBlue[block_idx]/255.0);
  }
}

// //kernel function to compute CDF, uses Brent-Kung algorithm from MP5.2
// __global__ void computeCDF(float *hist, float *CDF, int len){

//   __shared__ float partialSum[HISTOGRAM_LENGTH];
//   //set up shared mem
//   int i = 2*(blockIdx.x*blockDim.x + threadIdx.x);
  
//   if (i < len){
//     partialSum[2*threadIdx.x] = input[i];
//   }
//   else{
//     partialSum[2*threadIdx.x] = 0;
//   }

//   if (i+1 < len){
//     partialSum[2*threadIdx.x+1] = input[i+1];
//   }
//   else{
//     partialSum[2*threadIdx.x+1] = 0;
//   }
  
//   //Brent-Kung Algo
//   //Parallel Scan
//   int stride = 1;
//   int index = 0;
//   while(stride < HISTOGRAM_LENGTH) {
//     __syncthreads();
//     index = (threadIdx.x+1)*stride*2 - 1;
//     if(index < HISTOGRAM_LENGTH && (index-stride) >= 0)
//       partialSum[index] += partialSum[index-stride];
//     stride = stride*2;
//   }

//   //Post Scan
//   stride = HISTOGRAM_LENGTH/4;
//   while(stride > 0) {
//     __syncthreads();
//     index = (threadIdx.x+1)*stride*2 - 1;
//     if ((index+stride) < HISTOGRAM_LENGTH)
//       partialSum[index+stride] += partialSum[index];
//     stride = stride / 2;
//   }
//   //REQUIRED bc threads were writing to output before calculation was finished
//   __syncthreads();
  
//   //set output
//   if(i<len)
//     output[i] = partialSum[2*threadIdx.x];
  
//   if(i+1<len)
//    output[i+1] = partialSum[2*threadIdx.x+1];
// }

// __global__ void histEq2(float *input, float *output, float *histo, float* cdf, int height, int width){



// }

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
  int *sync_blocks;


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
  cudaMalloc((void**)&sync_blocks, sizeof(int));

  //copy input image to device
  cudaMemcpy(deviceInput, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);

  //set up grid dimensions
  dim3 DimGrid1(ceil((float)imageWidth/BLOCK_SIZE),ceil((float)imageHeight/BLOCK_SIZE),1);
  dim3 DimBlock1(BLOCK_SIZE,BLOCK_SIZE,1);

  // dim3 DimGrid2(1,1,1);
  // dim3 DimBlock2(HISTOGRAM_LENGTH/2,1,1);

  //call kernel
  histEq1<<<DimGrid1, DimBlock1>>>(deviceInput, deviceOutput, deviceHisto, deviceCDF, imageHeight, imageWidth, sync_blocks, HISTOGRAM_LENGTH);
  //computeCDF<<<DimGrid2, DimBlock2>>>(deviceHisto, deviceCDF, HISTOGRAM_LENGTH);
  //histEq2<<<DimGrid1, DimBlock1>>>(deviceInput, deviceOutput, deviceHisto, deviceCDF, imageHeight, imageWidth);


  //retrieve output from device
  cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth*imageHeight*sizeof(float), cudaMemcpyHostToDevice);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceHisto);
  cudaFree(deviceCDF);
  cudaFree(sync_blocks);

  return 0;
}
