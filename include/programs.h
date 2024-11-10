#pragma once

#include "operations.h"
#include <string>

namespace programs
{
    int vecadd(std::string&& size, Device device, Option& blocksize);
    int matmul(int dim, Device device, Option& blocksize);
    int color_to_gray(std::string&& quality, Device device, Option& blocksize);
    int stencil_1d(int radius, Device device, Option& blocksizeOpt);
}