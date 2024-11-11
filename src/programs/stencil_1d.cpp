#include "cli.h"
#include "operations.h"
#include "programs.h"
#include <format>
#include <iostream>

namespace programs
{
    int stencil_1d(int radius, Device device, Option& blocksizeOpt)
    {
        int size = 4096;
        int max_radius = size / 2 - 1;
        if (radius > max_radius)
        {
            radius = max_radius;
            std::cout << std::format("Capping to max radius {}\n", max_radius) << std::endl;
        }

        if (device == GPU && !blocksizeOpt.isSet())
        {
            std::cout << std::format("No blocksize specified. Default is {}\n", DEFAULT_BLOCK_SIZE)
                      << std::endl;
        }

        int blocksize = blocksizeOpt.getValue<int>();

        operations::stencil_1d(size, radius, device, blocksize);

        return 0;
    }
}