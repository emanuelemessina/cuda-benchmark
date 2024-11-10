#include "generate.h"
#include "kernels.cuh"
#include "operations.h"
#include "timer.h"
#include <format>
#include <vector>

namespace cpu
{
    void vecadd(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            c[i] = a[i] + b[i];
        }
    }
}

#pragma GCC push_options
#pragma GCC optimize("O0")

namespace operations
{
    void vecadd(int size, Device device, int gpuThreadsPerBlock = 256)
    {
        const std::vector<float> a = generate::random_vector(size), b = generate::random_vector(size);
        std::vector<float> c(size);

        if (device == GPU)
        {
            ScopedTimer execution(std::format("vecadd | GPU | {} [{}]", size, gpuThreadsPerBlock), PRE);
            cuda::vecadd(a.data(), b.data(), c.data(), size, gpuThreadsPerBlock);
        }
        else
        {
            ScopedTimer execution(std::format("vecadd | CPU | {}", size), PRE);
            cpu::vecadd(a, b, c, size);
        }
    }
}

#pragma GCC pop_options
