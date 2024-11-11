#include "kernels.cuh"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_global(const float* F, const float* K, float* C, int N, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
        return;

    float sum = 0.0f;

    for (int j = 0; j < k; ++j)
    {
        int idx = i - (k / 2 - j);

        if (idx >= 0 && idx < N)
        {
            sum += F[idx] * K[j];
        }
    }

    C[i] = sum;
}

namespace cuda
{
    void conv_1d_global(const float* F, const float* K, float* C, int N, int k, int blocksize)
    {
        int size = N * sizeof(float);

        float* d_F;
        float* d_K;
        float* d_C;
        cudaMalloc(&d_F, size);
        cudaMalloc(&d_K, k * sizeof(float));
        cudaMalloc(&d_C, size);

        {
            ScopedTimer t1("memcpy inputs CPU -> GPU", POST);
            cudaMemcpy(d_F, F, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_K, K, k * sizeof(float), cudaMemcpyHostToDevice);
        }

        {
            ScopedTimer t2("kernel execution", POST);
            int gridSize = (N + blocksize - 1) / blocksize;
            kernel_global<<<gridSize, blocksize>>>(d_F, d_K, d_C, N, k);
            CUDA_CHECK
            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        {
            ScopedTimer t3("memcpy output GPU -> CPU", POST);
            cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
        }

        cudaFree(d_F);
        cudaFree(d_K);
        cudaFree(d_C);
    }
}

// shared

__constant__ float d_K[4096];

__global__ void kernel_shared(const float* F, float* C, int N, int k, int pad)
{
    extern __shared__ float tile_F[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    int tdx = threadIdx.x + pad; // tile index

    // load tile
    tile_F[tdx] = F[idx]; // current thread

    // load halos (0 if oob)
    if (threadIdx.x < pad)
    {
        tile_F[tdx - pad] = (idx >= pad) ? F[idx - pad] : 0.0f;                         // load left halos
        tile_F[tdx + blockDim.x] = (idx + blockDim.x < N) ? F[idx + blockDim.x] : 0.0f; // load right halos
    }

    __syncthreads();

    float sum = 0.0f;

    for (int j = 0; j < k; ++j)
    {
        sum += tile_F[tdx - pad + j] * d_K[j];
    }

    C[idx] = sum;
}

namespace cuda
{
    void conv_1d_shared(const float* F, const float* K, float* C, int N, int k, int blocksize)
    {
        int pad = k / 2;
        int size = N * sizeof(float);

        float* d_F;
        float* d_C;
        cudaMalloc(&d_F, size);
        cudaMalloc(&d_C, size);

        {
            ScopedTimer t1("memcpy inputs CPU -> GPU", POST);
            cudaMemcpy(d_F, F, size, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(d_K, K, k * sizeof(float));
        }

        int gridSize = (N + blocksize - 1) / blocksize;
        int sharedMemSize = (blocksize + 2 * pad) * sizeof(float);

        {
            ScopedTimer t2("kernel execution", POST);
            kernel_shared<<<gridSize, blocksize, sharedMemSize>>>(d_F, d_C, N, k, pad);
            CUDA_CHECK
            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        {
            ScopedTimer t3("memcpy output GPU -> CPU", POST);
            cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_F);
        cudaFree(d_C);
    }
}
