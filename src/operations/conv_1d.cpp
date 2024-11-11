#include "generate.h"
#include "kernels.cuh"
#include "operations.h"
#include "timer.h"
#include <vector>

namespace cpu
{
    void conv_1d(const std::vector<float>& F, const std::vector<float>& K, std::vector<float> C, int size, int kernel)
    {
        for (int i = 0; i < size; ++i)
        {
            float sum = 0.0f;

            for (int j = 0; j < kernel; ++j)
            {
                int idx = i - (kernel / 2 - j);

                if (idx >= 0 && idx < size)
                {
                    sum += F[idx] * K[j];
                }
                // else sum += 0 * Kj
            }

            C[i] = sum;
        }
    }
}

#pragma GCC push_options
#pragma GCC optimize("O0")

namespace operations
{
    void conv_1d(int size, int kernel, Device device, int blocksize)
    {
        const auto F = generate::random_vector(size);
        const auto K = generate::random_vector(kernel);
        auto C = std::vector<float>(size);

        if (device == GPU)
        {
            {
                ScopedTimer execution(std::format("conv-1d | GPU (global) | {} : {} [{}]", size, kernel, blocksize), PRE);
                cuda::conv_1d_global(F.data(), K.data(), C.data(), size, kernel, blocksize);
            }
            {
                ScopedTimer execution(std::format("conv-1d | GPU (constant + shared) | {} : {} [{}]", size, kernel, blocksize), PRE);
                cuda::conv_1d_shared(F.data(), K.data(), C.data(), size, kernel, blocksize);
            }
        }
        else
        {
            ScopedTimer execution(std::format("conv-1d | CPU | {} : {}", size, kernel), PRE);
            cpu::conv_1d(F, K, C, size, kernel);
        }
    }

}

#pragma GCC pop_options
