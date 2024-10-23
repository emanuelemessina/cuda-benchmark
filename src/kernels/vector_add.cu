#include "kernels.cuh"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel(const float* a, const float* b, float* c, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

namespace cuda
{

    void vector_add(const float* ha, const float* hb, float* hc, size_t N, size_t threadsPerBlock)
    {

        float *da, *db, *dc;

        // Allocate memory on the device (GPU)
        cudaMalloc((void**)&da, N * sizeof(float));
        cudaMalloc((void**)&db, N * sizeof(float));
        cudaMalloc((void**)&dc, N * sizeof(float));

        {
            ScopedTimer t1("memcpy inputs CPU -> GPU", POST);

            // Copy vectors a and b from host to device
            cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice);
        }

        dim3 blockSize(threadsPerBlock, 0, 0);
        dim3 gridSize((N + threadsPerBlock - 1) / threadsPerBlock, 0, 0);

        {
            ScopedTimer t2("kernel execution", POST);

            kernel<<<gridSize, blockSize>>>(da, db, dc, N);

            // Wait for GPU to finish before accessing on host
            cudaDeviceSynchronize();
        }

        {
            ScopedTimer t3("memcpy output GPU -> CPU", POST);

            // Copy result vector c from device to host
            cudaMemcpy(hc, dc, N * sizeof(int), cudaMemcpyDeviceToHost);
        }

        // Free the allocated memory on the device
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
    }

}
