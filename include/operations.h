#pragma once

#include <array>
#include <cstddef>
#include <string>

#define DEFAULT_BLOCK_SIZE 32

inline static std::array<int, 9> blockSizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

enum Device
{
    CPU,
    GPU
};

namespace operations
{
    void vecadd(size_t size, Device device, size_t gpuThreadsPerBlock);
    void matmul(size_t size, Device device, size_t gpuThreadsPerBlock);
    void color_to_gray(const std::string& quality, Device device, int blocksize);
}
