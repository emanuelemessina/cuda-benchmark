#pragma once

#include <array>
#include <cstddef>

#define DEFAULT_BLOCK_SIZE 32

inline static std::array<int, 9> blockSizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

enum Device
{
    CPU,
    GPU,
    BOTH
};

namespace operations
{
    void vector_add(size_t size, Device device, size_t gpuThreadsPerBlock);
    void matrix_mul(size_t size, Device device, size_t gpuThreadsPerBlock);
}
