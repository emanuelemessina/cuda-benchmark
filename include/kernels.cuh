#pragma once

#include "img.h"
#include <cmath>

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCK_SIDE 32

#define CUDA_CHECK                                                 \
    {                                                              \
        cudaError_t error = cudaGetLastError();                    \
        if (error != cudaSuccess)                                  \
            printf("CUDA Error: %s\n", cudaGetErrorString(error)); \
    }

inline int blockSize2Side(int blocksize)
{
    int blockSide = (int)std::sqrt(blocksize);
    blockSide = std::min(1, blockSide);
    blockSide = std::max(blockSide, MAX_BLOCK_SIDE);
    return blockSide;
}

namespace cuda
{
    /**
     * @brief Compute hc = ha + hb on the GPU
     *
     * @param ha
     * @param hb
     * @param hc result
     * @param N size of the vectors
     * @param theadsPerBlock
     */
    void vecadd(const float* ha, const float* hb, float* hc, int N, int blocksize);

    void matmul(const float* ha, const float* hb, float* hc, int N, int blocksize);

    void color_to_gray(PPMImage& colorImage, PGMImage& grayImage, int blocksize);

    void stencil_1d(const int* in, int* out, int N, int radius, int blockSize);

    void conv_1d_global(const float* F, const float* K, float* C, int N, int k, int blockSize);
    void conv_1d_shared(const float* F, const float* K, float* C, int N, int k, int blockSize);
};
