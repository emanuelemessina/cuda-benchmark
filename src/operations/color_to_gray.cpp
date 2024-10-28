
#include "img.h"
#include "operations.h"
#include "timer.h"
#include <format>

namespace cpu
{
    void color_to_gray(PPMImage& colorImage, PGMImage& grayImage, int w, int h)
    {
        pixel** colorData = colorImage.getData();

        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                pixel& colorPixel = colorData[i][j];
                grayImage.data[i][j] = static_cast<unsigned char>(
                    0.299 * colorPixel.r + 0.587 * colorPixel.g + 0.114 * colorPixel.b);
            }
        }
    }
}

#pragma GCC push_options
#pragma GCC optimize("O0")

namespace operations
{
    void color_to_gray(const std::string& quality, Device device, int blocksize)
    {
        std::string filename = std::format("images/{}.ppm", quality);
        PPMImage colorImage = PPMImage{filename};

        int w = colorImage.getWidth();

        if (w == 0) // failed to load
            return;

        int h = colorImage.getHeight();

        PGMImage grayImage(w, h);

        if (device == GPU)
        {
            ScopedTimer execution(std::format("vecadd | GPU | {} [{}]", quality, blocksize), PRE);
            // cuda::vecadd(a.data(), b.data(), c.data(), size, gpuThreadsPerBlock);
        }
        else
        {
            ScopedTimer execution(std::format("vecadd | CPU | {}", quality), PRE);
            cpu::color_to_gray(colorImage, grayImage, w, h);
        }

        filename = std::format("images/{}_", quality);
        if (device == GPU)
        {
            filename.append(std::format("gpu_{}", blocksize));
        }
        else
        {
            filename.append("cpu");
        }

        filename.append(".pgm");

        grayImage.save(filename);
    }

}

#pragma GCC pop_options