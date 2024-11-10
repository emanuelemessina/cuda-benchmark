#include "kernels.cuh"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel(const int* in, int* out, int N, int radius)
{
    extern __shared__ int shared[];
    int* tile = shared;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < radius || idx >= N) // pad block position
        return;

    int tdx = threadIdx.x + radius; // current thread index in tile

    tile[tdx] = in[idx]; // central element (each thread)

    // halo elements for leftmost threads
    if (threadIdx.x < radius)
    {
        tile[threadIdx.x] = in[idx - radius]; // left

        if (idx + blockDim.x < N)                          // last block out of bounds
            tile[tdx + blockDim.x] = in[idx + blockDim.x]; // right
    }

    __syncthreads();

    int result = 0;
    for (int offset = -radius; offset <= radius; offset++)
    {
        result += tile[tdx + offset];
    }

    out[idx] = result;
}

namespace cuda
{
    void stencil_1d(const int* in, int* out, int N, int radius, int blockSize)
    {
        int *d_in, *d_out;

        int size = N * sizeof(int);

        cudaMalloc(&d_in, size);
        cudaMalloc(&d_out, size);

        {
            ScopedTimer t1("memcpy inputs CPU -> GPU", POST);
            cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);
        }

        {
            ScopedTimer t2("kernel execution", POST);
            kernel<<<(N + blockSize - 1) / blockSize, blockSize, (blockSize + 2 * radius) * sizeof(int)>>>(d_in, d_out, N, radius);
            CUDA_CHECK
            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        {
            ScopedTimer t3("memcpy output GPU -> CPU", POST);
            cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
        }

        // Error Checking
        for (int i = radius; i < N; i++)
        {
            if (i < radius || i >= N - radius)
            {
                // Halo elements should remain as 1
                if (out[i] != 1)
                    printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
            }
            else
            {
                // Stencil sum should be 1 + 2 * RADIUS (since input is initialized to 1)
                if (out[i] != 1 + 2 * radius)
                    printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2 * radius);
            }
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }
}