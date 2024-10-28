
#include "img.h"
#include "kernels.cuh"
#include "operations.h"
#include "timer.h"
#include <format>
#include <math.h>

namespace cpu
{
    void color_to_gray(PPMImage& colorImage, PGMImage& grayImage)
    {
        pixel* colorData = colorImage.getData();

        for (int i = 0; i < colorImage.numPixels(); ++i)
        {
            pixel& colorPixel = colorData[i];
            grayImage.data[i] = static_cast<unsigned char>(std::fminf(255.0f, 0.299 * colorPixel.r + 0.587 * colorPixel.g + 0.114 * colorPixel.b));
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
            ScopedTimer execution(std::format("color-to-gray | GPU | {} [{}]", quality, blocksize), PRE);
            cuda::color_to_gray(colorImage, grayImage, blocksize);
        }
        else
        {
            ScopedTimer execution(std::format("color-to-gray | CPU | {}", quality), PRE);
            cpu::color_to_gray(colorImage, grayImage);
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