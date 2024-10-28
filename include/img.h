#ifndef IMAGE_H
#define IMAGE_H

#include <fstream>
#include <iostream>
#include <string>

#define MAX_COLOR_DEPTH 255

struct pixel
{
    char r;
    char g;
    char b;
};

class Image
{
  protected:
    int width = 0;
    int height = 0;

  public:
    Image(const std::string& filename);
    Image(int w = 0, int h = 0);
    virtual ~Image();

    virtual bool load(const std::string& filename) = 0;
    virtual bool save(const std::string& filename) const = 0;

    int getWidth() const;
    int getHeight() const;
};

class PGMImage : public Image
{
  protected:
  public:
    PGMImage(int w = 0, int h = 0);
    ~PGMImage() override;

    bool load(const std::string& filename) override;
    bool save(const std::string& filename) const override;

    unsigned char** data;

  private:
    void skipComments(std::ifstream& file);
    void allocateData();
    void deallocateData();
};

class PPMImage : public Image
{
  protected:
    pixel** data = nullptr;

  public:
    PPMImage(const std::string& filename) { load(filename); }
    ~PPMImage() override;

    bool load(const std::string& filename) override;
    bool save(const std::string& filename) const override;
    pixel** getData() { return data; }

  private:
    void skipComments(std::ifstream& file);
    void allocateData();
    void deallocateData();
};

#endif // IMAGE_H
