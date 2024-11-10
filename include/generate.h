#include <algorithm>
#include <random>
#include <vector>

namespace generate
{
    inline std::vector<float> random_vector(int size)
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<float> distribution(1.f, 100.f);

        std::vector<float> vec(size);
        for (int i = 0; i < size; ++i)
        {
            vec[i] = distribution(generator);
        }

        return std::move(vec);
    }

    inline std::vector<std::vector<float>> random_matrix(int rows, int cols)
    {
        std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(1.0, 20.0);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                matrix[i][j] = dis(gen);
            }
        }

        return std::move(matrix);
    }

    inline std::vector<float> random_flat_matrix(int rows, int cols)
    {
        std::vector<float> matrix(rows * cols);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(1.0, 20.0);

        for (int i = 0; i < rows * cols; ++i)
        {
            matrix[i] = dis(gen);
        }

        return std::move(matrix);
    }

    inline std::vector<int> ones_vector(int size)
    {
        auto vec = std::vector<int>(size, 1);
        return std::move(vec);
    }
    inline std::vector<int> zero_vector(int size)
    {
        auto vec = std::vector<int>(size, 0);
        return std::move(vec);
    }
}
