
#include "generate.h"
#include "kernels.cuh"
#include "operations.h"
#include "timer.h"
#include <vector>

namespace cpu
{
    void stencil_1d(const std::vector<int>& in, std::vector<int>& out, int radius)
    {
        int size = in.size();
        for (int i = radius; i < size - radius; ++i)
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
    }
}

#pragma GCC push_options
#pragma GCC optimize("O0")

namespace operations
{
    void stencil_1d(int radius, Device device, int blocksize)
    {
        int N = 4096;
        const auto in = generate::ones_vector(N);
        auto out = generate::zero_vector(N);

        if (device == GPU)
        {
            ScopedTimer execution(std::format("stencil-1d | GPU | {} : {} [{}]", N, radius, blocksize), PRE);
            cuda::stencil_1d(in.data(), out.data(), N, radius, blocksize);
        }
        else
        {
            ScopedTimer execution(std::format("stencil-1d | CPU | {} : {}", N, radius), PRE);
            cpu::stencil_1d(in, out, radius);
        }
    }
}

#pragma GCC pop_options
