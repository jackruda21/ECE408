#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

// blockdim is this + K -1
#define TILE_WIDTH 21
#define TILE_HEIGHT 21
#define K_SIZE 7

/* optimizations:
 * 
 * TiledShared Mem - 2 pts
 * Matrix Multiplication/Unrolling - 3 pts
 * Streams - 4 pts
 *
 * Done:
 * Kernel in Constant Mem - 1 pt
 * Tuning unroll & restrict - 3pts
 * Fixed Point Arithmetic - 4pts
 */

int err_ct = 0;
__constant__ float KFILTER[1 * 7 * 7 * 4*16];

__global__ void conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) KFILTER[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(1.0*Width_out/TILE_WIDTH);
    //int H_grid = ceil(1.0*Width_out/TILE_WIDTH);

    int b, m, h, w;
    b = blockIdx.z;
    m = blockIdx.x;
    h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0;
    for (int c = 0; c < Channel; c++) { // sum over all input channels
        if(h< Height-K+1 && w < Width-K+1){
            #pragma unroll
            for (int p = 0; p < K_SIZE; p++){ // loop over KxK filter, unroll convolution
                for (int q = 0; q < K_SIZE; q++){
                        acc += in_4d(b, c, h+p, w+q) * mask_4d(m, c, p, q);
                }
            }
        }
    }
    if(h<Height_out && w<Width_out){
        out_4d(b,m,h,w) = acc;
    }
    
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_tiled(float* __restrict__ output, const /*__half* */ float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    //set up shared mem tile, Assume K is always 7, and that channel <= 4 
    __shared__ float input_tile[(TILE_HEIGHT+7-1)*(TILE_WIDTH+7-1)*4];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) KFILTER[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

        // Insert your GPU convolution kernel code here
    int W_grid = ceil(1.0*Width_out/TILE_WIDTH);
    //int H_grid = ceil(1.0*Width_out/TILE_WIDTH);

    int b, m;//, h, w;
    b = blockIdx.z;
    m = blockIdx.x;
    //h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    //w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    
    /*
    //tile variables, x,y of input
    int x = w - K/2;
    int y = h - K/2;
    //tile variables, local x,y threads w/ offset
    int tx = threadIdx.x - K/2;
    int ty = threadIdx.y - K/2;
    */

    /*__half*/ float acc = 0.0;

    int off = 3; // K/2

    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    int y = (blockIdx.y / W_grid) * TILE_HEIGHT + ty;
    int x = (blockIdx.y % W_grid) * TILE_WIDTH + tx;

    //HERE
    for (int c = 0; c < Channel; c++) { 
        //set tile
        input_tile[c*(TILE_HEIGHT+7-1)*(TILE_WIDTH+7-1) + ty*(TILE_WIDTH+K-1)+tx] = in_4d(b,c,y,x);
    }
    //sync
    __syncthreads();
    for (int c = 0; c < Channel; c++) { // sum over all input channels
        if(tx >= off && tx < (TILE_WIDTH+K-1-off) && ty >= off && ty < (TILE_HEIGHT+K-1-off)){
            //perform convolution
            #pragma unroll
            for (int p = 0; p < K; p++){ // loop over KxK filter
                for (int q = 0; q < K; q++){
                    //FP16
                    //acc = __hadd(acc, __hmul(input_tile[c*(TILE_HEIGHT+7-1)*(TILE_WIDTH+7-1) + (ty-off+p)*(TILE_WIDTH+K-1) + tx-off+q], __float2half(mask_4d(m, c, p, q))));
                    acc += input_tile[c*(TILE_HEIGHT+7-1)*(TILE_WIDTH+7-1) + (ty-off+p)*(TILE_WIDTH+K-1) + tx-off+q] * mask_4d(m, c, p, q);
                }
            }
        }
    }   

    if(tx >= off && tx < (TILE_WIDTH+off) && ty >= off && ty < (TILE_HEIGHT+off) && y>=off && y<Height_out+off && x>=off && x<Width_out+off){
        out_4d(b,m,(y-off),(x-off)) = acc;
    }



    /*
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    for (int c = 0; c < Channel; c++) { // sum over all input channels
        if(h< Height-K+1 && w < Width-K+1){
            #pragma unroll
            for (int p = 0; p < K_SIZE; p++){ // loop over KxK filter, unroll convolution
                for (int q = 0; q < K_SIZE; q++){
                        acc += in_4d(b, c, h+p, w+q) * mask_4d(m, c, p, q);
                }
            }
        }
    }
    if(h<Height_out && w<Width_out){
        out_4d(b,m,h,w) = acc;
    }*/
    
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_FP16(float* __restrict__ output, const __half* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    
    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) KFILTER[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(1.0*Width_out/TILE_WIDTH);
    //int H_grid = ceil(1.0*Width_out/TILE_WIDTH);

    int b, m;//, h, w;
    b = blockIdx.z;
    m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    __half acc = 0.0;
    for (int c = 0; c < Channel; c++) { // sum over all input channels
        if(h< Height-K+1 && w < Width-K+1){
            #pragma unroll
            for (int p = 0; p < K_SIZE; p++){ // loop over KxK filter, unroll convolution
                for (int q = 0; q < K_SIZE; q++){
                    acc = __hadd(acc, __hmul(in_4d(b, c, h+p, w+q), __float2half(mask_4d(m, c, p, q))));
                }
            }
        }
    }
    if(h<Height_out && w<Width_out){
        out_4d(b,m,h,w) = __half2float(acc);
    }
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

#define CONVERT_SIZE 1024

__global__ void convert_to_half(const float* input, __half* output, const int size){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, 
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    //calculate input/output sizes
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int input_size = Height*Width*Channel*Batch;
    int output_size = Height_out*Width_out*Map_out*Batch;
    //int mask_size = K*K*Channel*Map_out;

    cudaMalloc((void**)device_input_ptr, input_size * sizeof(float));
    //std::cout<<"Size: "<<input_size * sizeof(float)<<'\n';
    cudaMalloc((void**)device_output_ptr, output_size * sizeof(float));
    //cudaMalloc((void**)device_mask_ptr, mask_size * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(*device_mask_ptr, host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_output_ptr, host_output, output_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(KFILTER, host_mask, 1 * 7 * 7 * 4*16*sizeof(float));

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        err_ct +=1;
        std::cout<<"\t\tCUDA error: "<<cudaGetErrorString(error)<<" -- "<<err_ct<<std::endl;
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, 
                                             const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    
    //std::cout<<"K: "<<K<<'\n';
    // std::cout<<"H: "<<Height<<'\n';
    // std::cout<<"W: "<<Width<<'\n';
    // std::cout<<"C: "<<Channel<<'\n';

    //FP16
    /*__half* device_half_input;
    int input_size = Height*Width*Channel*Batch;

    cudaMalloc((void**)&device_half_input, input_size * sizeof(__half));
    dim3 blockConv(CONVERT_SIZE,1,1);
    dim3 gridConv(ceil(1.0*input_size/CONVERT_SIZE),1,1);
    convert_to_half<<<gridConv, blockConv>>>(device_input, device_half_input, input_size);*/
    

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int W_grid = ceil(1.0*Width_out/TILE_WIDTH); // number of horizontal tiles per output map
    int H_grid = ceil(1.0*Height_out/TILE_HEIGHT); // number of vertical tiles per output map
    int Y = H_grid * W_grid;

    dim3 blockDim(TILE_WIDTH+K-1, TILE_HEIGHT+K-1, 1); // output tile for untiled code
    dim3 gridDim(Map_out, Y, Batch);

    // std::cout<<"Block Size: "<<(TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)<<'\n';

    //FP16
    /*conv_forward_kernel_FP16<<<gridDim, blockDim >>>(device_output, device_half_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();

    cudaFree(device_half_input);*/

    conv_forward_kernel<<<gridDim, blockDim >>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();



}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int output_size = Height_out*Width_out*Map_out*Batch;

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        err_ct += 10;
        std::cout<<"\t\tCUDA error: "<<cudaGetErrorString(error)<<" -- "<<err_ct<<std::endl;
    }
    
    cudaMemcpy(host_output, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree((__half*)device_input);
    cudaFree(device_output);
    //cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
