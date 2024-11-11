#include "cli.h"
#include "operations.h"
#include "programs.h"

namespace programs
{
    int conv_1d(int size, int kernel, Device device, Option& blocksizeOpt)
    {
        clamp_int_argument(size, 32, 8192);
        clamp_int_argument(kernel, 1, size / 2 - 1);

        int blocksize = get_blocksize_or_alert_default(device, blocksizeOpt);

        operations::conv_1d(size, kernel, device, blocksize);

        return 0;
    }
}