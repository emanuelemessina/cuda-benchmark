#ifndef IMAGE_H
#define IMAGE_H

#include <fstream>
#include <iostream>
#include <string>

#define MAX_COLOR_DEPTH 255

struct pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

class Image
{
  protected:
    int width = 0;
    int height = 0;

  public:
    Image() = default;
    Image(int w, int h) : width(w), height(h) {}
    ~Image() = default;

    virtual bool load(const std::string& filename) = 0;
    virtual bool save(const std::string& filename) const = 0;

    int getWidth() const { return width; };
    int getHeight() const { return height; };
    int numPixels() const { return width * height; };
};

class PGMImage : public Image
{
  protected:
  public:
    PGMImage(int w = 0, int h = 0) : Image(w, h)
    {
        allocateData();
    }
    ~PGMImage() { delete[] data; }

    bool load(const std::string& filename) override;
    bool save(const std::string& filename) const override;

    unsigned char* data;

  private:
    void allocateData() { data = new unsigned char[numPixels()]; }
};

class PPMImage : public Image
{
  protected:
    pixel* data = nullptr;

  public:
    PPMImage(const std::string& filename) { load(filename); }
    ~PPMImage() { delete[] data; };

    bool load(const std::string& filename) override;
    bool save(const std::string& filename) const override;
    pixel* getData() { return data; }

  private:
    void allocateData() { data = new pixel[numPixels()]; }
};

#endif // IMAGE_H
