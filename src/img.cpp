#include "img.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

Image::Image(int w, int h) : width(w), height(h) {}
Image::~Image() = default;

int Image::getWidth() const { return width; }
int Image::getHeight() const { return height; }

PGMImage::PGMImage(int w, int h) : Image(w, h)
{
    allocateData();
}

PGMImage::~PGMImage()
{
    deallocateData();
}

bool PGMImage::save(const std::string& filename) const
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Unable to open file '" << filename << "'\n";
        return false;
    }

    file << "P5\n"
         << width << " " << height << "\n"
         << MAX_COLOR_DEPTH << "\n";

    for (int i = 0; i < height; ++i)
    {
        file.write(reinterpret_cast<const char*>(data[i]), width);
    }

    return file.good();
}

bool PGMImage::load(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Unable to open file '" << filename << "'\n";
        return false;
    }

    std::string magic;
    file >> magic;
    if (magic != "P5")
    {
        std::cerr << "Invalid PGM format (must be 'P5')\n";
        return false;
    }

    file.ignore();
    skipComments(file);
    file >> width >> height;

    int depth;
    file >> depth;

    if (depth != MAX_COLOR_DEPTH)
    {
        std::cerr << "'" << filename << "' does not have 8-bits components\n";
        return false;
    }

    file.ignore();
    allocateData();

    for (int i = 0; i < height; ++i)
    {
        file.read(reinterpret_cast<char*>(data[i]), width);
    }

    return file.good();
}

void PGMImage::skipComments(std::ifstream& file)
{
    char c;
    while ((c = file.peek()) == '#')
    {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

void PGMImage::allocateData()
{
    data = new unsigned char*[height];
    for (int i = 0; i < height; ++i)
    {
        data[i] = new unsigned char[width];
    }
}

void PGMImage::deallocateData()
{
    for (int i = 0; i < height; ++i)
    {
        delete[] data[i];
    }
    delete[] data;
}

PPMImage::~PPMImage()
{
    deallocateData();
}

bool PPMImage::load(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Unable to open file '" << filename << "'\n";
        return false;
    }

    std::string magic;
    file >> magic;
    if (magic != "P6")
    {
        std::cerr << "Invalid PPM format (must be 'P6')\n";
        return false;
    }

    file.ignore();
    skipComments(file);

    int depth;
    file >> width >> height >> std::ws >> depth;

    if (depth != MAX_COLOR_DEPTH)
    {
        std::cerr << "'" << filename << "' does not have 8-bits components\n";
        return false;
    }

    file.ignore();
    allocateData();

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            file.read(&data[i][j].r, 1);
            file.read(&data[i][j].g, 1);
            file.read(&data[i][j].b, 1);
        }
    }

    return file.good();
}

bool PPMImage::save(const std::string& filename) const
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Unable to open file '" << filename << "'\n";
        return false;
    }

    file << "P6\n"
         << width << " " << height << "\n"
         << MAX_COLOR_DEPTH << "\n";

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            file.write(&data[i][j].r, 1);
            file.write(&data[i][j].g, 1);
            file.write(&data[i][j].b, 1);
        }
    }

    return file.good();
}

void PPMImage::skipComments(std::ifstream& file)
{
    char c;
    while ((c = file.peek()) == '#')
    {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

void PPMImage::allocateData()
{
    data = new pixel*[height];
    for (int i = 0; i < height; ++i)
    {
        data[i] = new pixel[width];
    }
}

void PPMImage::deallocateData()
{
    for (int i = 0; i < height; ++i)
    {
        delete[] data[i];
    }
    delete[] data;
}
