
#include "cli.h"
#include "programs.h"
#include <algorithm>
#include <format>
#include <iostream>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace programs
{
    int vecadd(std::string&& sizeStr, Device device, Option& blocksizeOpt)
    {
        std::unordered_map<std::string, size_t>
            vectorSizes = {
                {"1k", 1000},
                {"10k", 10000},
                {"1M", 1000000}};

        auto valuesRange = vectorSizes | std::views::values;
        std::vector<size_t> sizes(valuesRange.begin(), valuesRange.end());

        std::ranges::sort(sizes);

        if (sizeStr.empty())
        {
            std::cout << "No vector size specified. Trying all vector sizes\n"
                      << std::endl;
        }
        else if (vectorSizes.find(sizeStr) != vectorSizes.end())
        {
            size_t size = vectorSizes[sizeStr];
            sizes = {size};

            std::cout << "Selected vector size: " << sizeStr << "\n"
                      << std::endl;
        }
        else
        {
            std::cerr << "Invalid vector size option. Use: ";
            for (const auto& pair : vectorSizes)
            {
                std::cout << pair.first << " ";
            }
            std::cout << std::endl;

            return 1;
        }

        bool special10kTest = !blocksizeOpt.isSet();

        if (device == GPU && !blocksizeOpt.isSet())
        {
            std::cout << std::format("No blocksize specified. Default is {}\n", DEFAULT_BLOCK_SIZE)
                      << std::endl;
        }

        int blocksize = blocksizeOpt.getValue<int>();

        for (const auto& size : sizes)
        {
            if (size == 10000 && device == GPU && special10kTest)
            {
                std::cout << "Special 10k test: trying all blocksizes for N = 10k\n"
                          << std::endl;

                for (int bs : blockSizes)
                {
                    operations::vecadd(size, device, bs);
                }

                continue;
            }

            operations::vecadd(size, device, blocksize);
        }

        return 0;
    }
}
