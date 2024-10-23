#include "cli.h"
#include "operations.h"
#include "programs.h"
#include <iostream>
#include <map>
#include <ranges>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    // cli definition

    CLI cli = CLI{"Vector addition and Matrix multiplication tests"};

    cli
        .option({"v", "vecadd", OPTION_STRING_UNSET})
        .option({"m", "matmul", OPTION_INT_UNSET})
        .option({"d", "device", OPTION_STRING_UNSET})
        .option({"b", "blocksize", DEFAULT_BLOCK_SIZE});

    cli.parse(argc, argv);

    // common options

    Device device = BOTH;
    auto deviceOpt = cli.get("device");
    if (deviceOpt.isSet())
    {
        device = deviceOpt.getValue<std::string>() == "gpu" ? GPU : CPU;
    }

    auto blocksize = cli.get("blocksize");

    if (device == GPU && blocksize.isSet())
    {
        if (std::find(blockSizes.begin(), blockSizes.end(), blocksize.getValue<int>()) == blockSizes.end())
        {
            std::cerr << "Invalid block size option. Use: ";
            for (int bs : blockSizes)
            {
                std::cout << bs << " ";
            }
            std::cout << std::endl;
            return 1;
        }
    }

    // programs

    auto vecadd = cli.get("vecadd");

    if (vecadd.isSet())
    {
        auto size = vecadd.getValue<std::string>();

        if (device == BOTH)
        {
            std::cout << "No device selected. Trying both CPU and GPU\n"
                      << std::endl;
            int result = programs::vecadd(std::move(size), CPU, blocksize);
            result |= programs::vecadd(std::move(size), GPU, blocksize);
            return result;
        }

        return programs::vecadd(std::move(size), device, blocksize);
    }

    auto matmul = cli.get("matmul");

    if (matmul.isSet())
    {
        int dim = matmul.getValue<int>();

        if (device == BOTH)
        {
            std::cout << "No device selected. Trying both CPU and GPU\n"
                      << std::endl;
            int result = programs::matmul(dim, CPU, blocksize);
            result |= programs::matmul(dim, GPU, blocksize);
            return result;
        }

        return programs::matmul(dim, device, blocksize);
    }

    // help

    cli.help();

    return 0;
}