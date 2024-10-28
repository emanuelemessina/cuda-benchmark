#include "img.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

void skipComments(std::ifstream& file)
{
    char c;
    while ((c = file.peek()) == '#')
    {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
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

    file.write(reinterpret_cast<const char*>(data), numPixels());

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

    file.read(reinterpret_cast<char*>(data), numPixels());

    return file.good();
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

    file.read(reinterpret_cast<char*>(data), numPixels() * 3);

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

    file.write(reinterpret_cast<const char*>(data), numPixels() * 3); // Write the entire data block

    return file.good();
}
