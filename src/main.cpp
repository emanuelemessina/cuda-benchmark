#include "cli.h"
#include "operations.h"
#include "programs.h"
#include <iostream>
#include <map>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    // cli definition

    CLI cli = CLI{"CUDA Benchmark"};

    cli
        .option({"v", "vecadd", OPTION_STRING_UNSET, "Vector addition"})
        .option({"m", "matmul", OPTION_INT_UNSET, "Matrix multiplication"})
        .option({"c2g", "color-to-gray", OPTION_STRING_UNSET, "RBG to Grayscale conversion"})
        .option({"s1d", "stencil-1d", 3, "Stencil 1D"})
        .option({"d", "device", OPTION_STRING_UNSET, "gpu|cpu"})
        .option({"b", "blocksize", DEFAULT_BLOCK_SIZE});

    cli.parse(argc, argv);

    // common options

    std::vector<Device> devices = {CPU, GPU};
    auto deviceOpt = cli.get("device");
    if (deviceOpt.isSet())
    {
        devices[0] = deviceOpt.getValue<std::string>() == "gpu" ? GPU : CPU;
        devices.pop_back();
    }

    if (devices.size() > 1)
    {
        std::cout << "No device selected. Trying both CPU and GPU\n"
                  << std::endl;
    }

    int result = 0;

    for (Device device : devices)
    {

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
            result |= programs::vecadd(std::move(size), device, blocksize);
            continue;
        }

        auto matmul = cli.get("matmul");

        if (matmul.isSet())
        {
            int dim = matmul.getValue<int>();
            result |= programs::matmul(dim, device, blocksize);
            continue;
        }

        auto c2g = cli.get("color-to-gray");

        if (c2g.isSet())
        {
            std::string q = c2g.getValue<std::string>();
            result |= programs::color_to_gray(std::move(q), device, blocksize);
            continue;
        }

        auto stencil_1d = cli.get("stencil-1d");

        if (stencil_1d.isSet())
        {
            int radius = stencil_1d.getValue<int>();
            result |= programs::stencil_1d(radius, device, blocksize);
            continue;
        }

        // help

        cli.help();
        return 0;
    }
}