#include <random>
#include <vector>

inline std::vector<float> generate_random_vector(size_t size)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(1.f, 100.f);

    std::vector<float> vec(size);
    for (size_t i = 0; i < size; ++i)
    {
        vec[i] = distribution(generator);
    }

    return std::move(vec);
}

inline std::vector<std::vector<float>> generate_random_matrix(int rows, int cols)
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

inline std::vector<float> generate_random_flat_matrix(int rows, int cols)
{
    std::vector<float> matrix(rows * cols);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0, 20.0);

    for (int i = 0; i < rows * cols; ++i)
    {
        matrix[i] = dis(gen);
    }

    return matrix;
}
