#include "cli.h"
#include "programs.h"
#include <algorithm>
#include <format>
#include <iostream>
#include <vector>

namespace programs
{
    int matmul(int dim, Device device, Option& blocksizeOpt)
    {
        std::vector<int> matrixDimensions = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

        if (dim == -1)
        {
            std::cout << "Will try all matrix dimensions\n"
                      << std::endl;
        }
        else if (std::find(matrixDimensions.begin(), matrixDimensions.end(), dim) != matrixDimensions.end())
        {
            matrixDimensions = {dim};

            std::cout << "Selected matrix dimension: " << dim << "\n"
                      << std::endl;
        }
        else
        {
            std::cerr << "Invalid matrix dimension option. Use: ";
            for (int dim : matrixDimensions)
            {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            return 1;
        }

        bool special4046Test = !blocksizeOpt.isSet();

        if (!blocksizeOpt.isSet())
        {
            std::cout << std::format("No blocksize specified. Default is {}\n", DEFAULT_BLOCK_SIZE)
                      << std::endl;
        }

        int blocksize = blocksizeOpt.getValue<int>();

        for (const auto& dimension : matrixDimensions)
        {
            if (dimension == 4096 && device == GPU && special4046Test)
            {
                std::cout << "Special 4096 test: trying all blocksizes for N = 4096\n"
                          << std::endl;

                for (int bs : blockSizes)
                {
                    operations::matrix_mul(dimension, device, bs);
                }

                continue;
            }

            operations::matrix_mul(dimension, device, blocksize);
        }

        return 0;
    }
}
