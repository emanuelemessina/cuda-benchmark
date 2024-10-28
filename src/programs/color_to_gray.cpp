#include "cli.h"
#include "img.h"
#include "operations.h"
#include <algorithm>
#include <format>
#include <vector>

namespace programs
{
    int color_to_gray(std::string&& quality, Device device, Option& blocksizeOpt)
    {
        std::vector<std::string> qualities = {"4k", "fhd", "hd", "sd", "32"};

        if (quality.empty())
        {
            std::cout << "No image quality specified. Trying all qualities\n"
                      << std::endl;
        }
        else if (std::find(qualities.begin(), qualities.end(), quality) != qualities.end())
        {
            qualities = {quality};

            std::cout << "Selected quality: " << quality << "\n"
                      << std::endl;
        }
        else
        {
            std::cerr << "Invalid quality option. Use: ";
            for (const auto& q : qualities)
            {
                std::cout << q << " ";
            }
            std::cout << std::endl;

            return 1;
        }

        //

        if (device == GPU && !blocksizeOpt.isSet())
        {
            std::cout << std::format("No blocksize specified. Default is {}\n", DEFAULT_BLOCK_SIZE)
                      << std::endl;
        }

        int blocksize = blocksizeOpt.getValue<int>();

        //

        for (const auto& q : qualities)
        {
            operations::color_to_gray(q, device, blocksize);
        }

        return 0;
    }

}