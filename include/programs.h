#pragma once

#include "operations.h"
#include <string>

namespace programs
{
    int vecadd(std::string&& size, Device device, Option& blocksize);
    int matmul(int dim, Device device, Option& blocksize);
    int color_to_gray(std::string&& quality, Device device, Option& blocksize);
    int stencil_1d(int size, int radius, Device device, Option& blocksizeOpt);
    int conv_1d(int size, int kernel, Device device, Option& blocksizeOpt);
}

inline int get_blocksize_or_alert_default(Device device, Option& blocksizeOpt)
{
    if (device == GPU && !blocksizeOpt.isSet())
    {
        std::cout << std::format("No blocksize specified. Default is {}\n", DEFAULT_BLOCK_SIZE)
                  << std::endl;
    }

    return blocksizeOpt.getValue<int>();
}