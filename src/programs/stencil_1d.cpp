#include "cli.h"
#include "operations.h"
#include "programs.h"
#include <format>
#include <iostream>

namespace programs
{
    int stencil_1d(int radius, Device device, Option& blocksizeOpt)
    {
        if (device == GPU && !blocksizeOpt.isSet())
        {
            std::cout << std::format("No blocksize specified. Default is {}\n", DEFAULT_BLOCK_SIZE)
                      << std::endl;
        }

        int blocksize = blocksizeOpt.getValue<int>();

        operations::stencil_1d(radius, device, blocksize);

        return 0;
    }
}