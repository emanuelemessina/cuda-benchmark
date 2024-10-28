#include "generate.h"
#include "kernels.cuh"
#include "operations.h"
#include "timer.h"
#include <format>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace cpu
{
    void matmul(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& c,
                    size_t size)
    {
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                for (int k = 0; k < size; ++k)
                {
                    c[i * size + j] += a[i * size + k] * b[k * size + j];
                }
            }
        }
    }
}

const float* flattenMatrix(const std::vector<std::vector<float>>& matrix2D)
{
    size_t rows = matrix2D.size();
    size_t cols = matrix2D[0].size();

    float* flat = new float[rows * cols];

    int idx = 0;
    for (const auto& row : matrix2D)
    {
        for (float val : row)
        {
            flat[idx++] = val;
        }
    }

    return flat;
}

void printMatrix(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << std::setw(5) << std::fixed << std::setprecision(0)
                      << matrix[i * cols + j]; // Access and format element
        }
        std::cout << std::endl; // New line after each row
    }
}

#pragma GCC push_options
#pragma GCC optimize("O0")

namespace operations
{
    void matmul(size_t size, Device device, size_t gpuThreadsPerBlock)
    {
        const std::vector<float> a = generate_random_flat_matrix(size, size), b = generate_random_flat_matrix(size, size);
        std::vector<float> c(size * size);

        if (device == GPU)
        {
            float* flat_c = new float[size * size];
            ScopedTimer execution(std::format("matmul | GPU | {} [{}]", size, gpuThreadsPerBlock), PRE);
            cuda::matmul(a.data(), b.data(), c.data(), size, gpuThreadsPerBlock);
        }
        else
        {
            ScopedTimer execution(std::format("matmul | CPU | {}", size), PRE);
            cpu::matmul(a, b, c, size);
        }
    }
}

#pragma GCC pop_options
