#pragma once

#include <cuda_runtime.h>
#include <math.h>

inline __host__ __device__ unsigned char depthCap(float pixelValue)
{
    return static_cast<unsigned char>(std::fminf(pixelValue, MAX_COLOR_DEPTH));
}

inline __host__ __device__ unsigned char toGrayscale(const pixel& p)
{
    return depthCap(0.299 * p.r + 0.587 * p.g + 0.114 * p.b);
}