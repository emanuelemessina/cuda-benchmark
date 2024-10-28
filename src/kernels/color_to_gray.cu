#include "img.h"
#include "kernels.cuh"
#include "pixel.cuh"
#include "timer.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

__global__ void kernel(const pixel* colorData, unsigned char* grayData, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * width + x;

    if (x < width && y < height)
    {
        const pixel& colorPixel = colorData[idx];
        grayData[idx] = toGrayscale(colorPixel);
    }
}

namespace cuda
{
    void color_to_gray(PPMImage& colorImage, PGMImage& grayImage, int blocksize)
    {
        int numPixels = colorImage.numPixels();
        int width = colorImage.getWidth();
        int height = colorImage.getHeight();

        pixel* d_colorData;
        unsigned char* d_grayData;

        cudaMalloc(&d_colorData, numPixels * sizeof(pixel));
        cudaMalloc(&d_grayData, numPixels * sizeof(unsigned char));

        pixel* h_linearColorData = colorImage.getData();
        {
            ScopedTimer t1("memcpy input CPU -> GPU", POST);
            cudaMemcpy(d_colorData, h_linearColorData, numPixels * sizeof(pixel), cudaMemcpyHostToDevice);
        }

        int blockSide = blockSize2Side(blocksize);
        dim3 threadsPerBlock(blockSide, blockSide);
        dim3 blocksPerGrid((width + blockSide - 1) / blockSide, (height + blockSide - 1) / blockSide);

        {
            ScopedTimer t2("kernel execution", POST);
            kernel<<<blocksPerGrid, threadsPerBlock>>>(d_colorData, d_grayData, width, height);
            CUDA_CHECK
            cudaDeviceSynchronize();
        }

        {
            ScopedTimer t3("memcpy output GPU -> CPU", POST);
            cudaMemcpy(grayImage.data, d_grayData, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_colorData);
        cudaFree(d_grayData);
    }
}