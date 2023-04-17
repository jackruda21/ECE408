#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLOCK_SIZE 64

__global__ void spmvJDSKernel(float *out, int *matColStart, int *matCols,
                              int *matRowPerm, int *matRows,
                              float *matData, float *vec, int dim) {
  //@@ insert spmv kernel for jds format
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < dim) {
    float dot = 0;
    int num_in_row = matRows[row];
    int col_start = 0;
    int num_count = 0;
    //for (int i = 0; i< num_in_row; i++) {
      //matColStart[i] would be the column in which i is in
      //but maybe you don't have an el in every column, sou you'd
      //have to increment i again which is why we use col_start instead

      //now if the column index position would be into the next column
      //based on the start indices, we know that there shouldn't be
      //any data in this column so we increment col_start until we get a valid col.

      //A
      //works on even less rows than B & C
      // while(matColStart[col_start]+row >= matColStart[col_start+1]){
      //   col_start ++;
      // }


      //B
      //works equally as well as C
      /*
      for(int i=0; i<matRows[0]; i++){
        if(matColStart[i] + row < matColStart[i+1]){
          dot += matData[matColStart[i]+row] * vec[matCols[matColStart[i]+row]];  
        }
      }*/


      //C
      //works except for a few rows for some reason
      /*
      while(matColStart[col_start+1] - matColStart[col_start] > row){
        dot += matData[matColStart[col_start]+row] * vec[matCols[matColStart[col_start]+row]];
        col_start ++;
      }*/
    //}
      
    //after many failed attempts, THIS loop works
    //what went wrong: logic was correct, but when we incremented 1 past the end of matColStart,
    //our logic would no longer hold b/c it would be reading (assumadly) 0 for matColStart[end + 1]
    //now we know that if there are still more row elements, and that we are in the last section of 
    //matColStart, then we KNOW that this is the last element and that we won't be going past
    //matColStart[col_start+1] since there is no matColStart[col_start+1]
    //So if col_start = max index of matColStart, auto-pass check. matRows[0] is the row w/ 
    //the larges # elements, so that -1 is max index of matColStart
    while(num_count < num_in_row){
      if(matColStart[col_start] + row < matColStart[col_start+1] || col_start == matRows[0]-1){
        dot += matData[matColStart[col_start]+row] * vec[matCols[matColStart[col_start]+row]];
        num_count ++;
      }
      col_start ++;
      if(col_start == 1000){
        break;
      }
    }
    out[matRowPerm[row]] = dot;
  }
}

static void spmvJDS(float *out, int *matColStart, int *matCols,
                    int *matRowPerm, int *matRows, float *matData,
                    float *vec, int dim) {

  //@@ invoke spmv kernel for jds format
  //set up gridsize
  dim3 DimGrid(ceil((float)dim/BLOCK_SIZE),1,1);
  dim3 DimBlock(BLOCK_SIZE,1,1);

  //call kernel
  spmvJDSKernel<<<DimGrid, DimBlock>>>(out, matColStart, matCols, matRowPerm, matRows, matData, vec, dim);

}

int main(int argc, char **argv) {
  wbArg_t args;
  int *hostCSRCols;
  int *hostCSRRows;
  float *hostCSRData;
  int *hostJDSColStart;
  int *hostJDSCols;
  int *hostJDSRowPerm;
  int *hostJDSRows;
  float *hostJDSData;
  float *hostVector;
  float *hostOutput;
  int *deviceJDSColStart;
  int *deviceJDSCols;
  int *deviceJDSRowPerm;
  int *deviceJDSRows;
  float *deviceJDSData;
  float *deviceVector;
  float *deviceOutput;
  int dim, ncols, nrows, ndata;
  int maxRowNNZ;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostCSRCols = (int *)wbImport(wbArg_getInputFile(args, 0), &ncols, "Integer");
  hostCSRRows = (int *)wbImport(wbArg_getInputFile(args, 1), &nrows, "Integer");
  hostCSRData = (float *)wbImport(wbArg_getInputFile(args, 2), &ndata, "Real");
  hostVector = (float *)wbImport(wbArg_getInputFile(args, 3), &dim, "Real");

  hostOutput = (float *)malloc(sizeof(float) * dim);

  wbTime_stop(Generic, "Importing data and creating memory on host");

  CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm, &hostJDSRows,
           &hostJDSColStart, &hostJDSCols, &hostJDSData);
  maxRowNNZ = hostJDSRows[0];

  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceJDSColStart, sizeof(int) * maxRowNNZ);
  cudaMalloc((void **)&deviceJDSCols, sizeof(int) * ndata);
  cudaMalloc((void **)&deviceJDSRowPerm, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSRows, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSData, sizeof(float) * ndata);

  cudaMalloc((void **)&deviceVector, sizeof(float) * dim);
  cudaMalloc((void **)&deviceOutput, sizeof(float) * dim);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm, deviceJDSRows,
          deviceJDSData, deviceVector, dim);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceVector);
  cudaFree(deviceOutput);
  cudaFree(deviceJDSColStart);
  cudaFree(deviceJDSCols);
  cudaFree(deviceJDSRowPerm);
  cudaFree(deviceJDSRows);
  cudaFree(deviceJDSData);

  wbTime_stop(GPU, "Freeing GPU Memory");

  //DEBUGGING CODE
  /*
  if(dim == 16){
    wbLog(TRACE, "JDSColStart:");
    for(int i=0; i<maxRowNNZ; i++){
      wbLog(TRACE, hostJDSColStart[i]);
    }

    wbLog(TRACE, "JDSCols:");
    for(int i=0; i<ndata; i++){
      wbLog(TRACE, hostJDSCols[i]);
    }

    wbLog(TRACE, "JDSRowPerm:");
    for(int i=0; i<dim; i++){
      wbLog(TRACE, hostJDSRowPerm[i]);
    }

    wbLog(TRACE, "JDSRows:");
    for(int i=0; i<dim; i++){
      wbLog(TRACE, hostJDSRows[i]);
    }

    wbLog(TRACE, "JDSData:");
    for(int i=0; i<ndata; i++){
      wbLog(TRACE, hostJDSData[i]);
    }

    wbLog(TRACE, "Output:");
    for(int i=0; i<dim; i++){
      wbLog(TRACE, hostOutput[i]);
    }
  }*/

  wbSolution(args, hostOutput, dim);

  free(hostCSRCols);
  free(hostCSRRows);
  free(hostCSRData);
  free(hostVector);
  free(hostOutput);
  free(hostJDSColStart);
  free(hostJDSCols);
  free(hostJDSRowPerm);
  free(hostJDSRows);
  free(hostJDSData);

  return 0;
}
