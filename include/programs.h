#pragma once

#include "operations.h"
#include <string>

namespace programs
{
    int vecadd(std::string&& size, Device device, Option& blocksize);
    int matmul(int dim, Device device, Option& blocksize);
}