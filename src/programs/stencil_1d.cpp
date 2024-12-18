#include "cli.h"
#include "operations.h"
#include "programs.h"
#include <format>
#include <iostream>

namespace programs
{
    int stencil_1d(int size, int radius, Device device, Option& blocksizeOpt)
    {
        clamp_int_argument(size, 32, 8192);
        clamp_int_argument(radius, 1, size / 2 - 1);

        int blocksize = get_blocksize_or_alert_default(device, blocksizeOpt);

        operations::stencil_1d(size, radius, device, blocksize);

        return 0;
    }
}