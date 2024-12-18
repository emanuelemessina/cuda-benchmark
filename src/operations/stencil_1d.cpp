
#include "generate.h"
#include "kernels.cuh"
#include "operations.h"
#include "timer.h"
#include <vector>

namespace cpu
{
    void stencil_1d(const std::vector<int>& in, std::vector<int>& out, int size, int radius)
    {
        for (int i = radius; i < size - radius; ++i)
        {
            if (i >= radius && i < size - radius)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    int idx = i + j;
                    if (idx < 0)
                        continue;
                    if (idx >= size)
                        break;
                    out[i] += in[i + j];
                }
            }
            else
            {
                out[i] = in[i];
            }
        }
    }
}

#pragma GCC push_options
#pragma GCC optimize("O0")

namespace operations
{
    void stencil_1d(int N, int radius, Device device, int blocksize)
    {
        const auto in = generate::ones_vector(N);
        auto out = std::vector<int>(N);

        if (device == GPU)
        {
            ScopedTimer execution(std::format("stencil-1d | GPU | {} : {} [{}]", N, radius, blocksize), PRE);
            cuda::stencil_1d(in.data(), out.data(), N, radius, blocksize);
        }
        else
        {
            ScopedTimer execution(std::format("stencil-1d | CPU | {} : {}", N, radius), PRE);
            cpu::stencil_1d(in, out, N, radius);
        }
    }
}

#pragma GCC pop_options
