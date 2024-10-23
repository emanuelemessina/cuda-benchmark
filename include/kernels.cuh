#pragma once

namespace cuda
{
    /**
     * @brief Compute hc = ha + hb on the GPU
     *
     * @param ha
     * @param hb
     * @param hc result
     * @param N size of the vectors
     * @param theadsPerBlock
     */
    void vector_add(const float* ha, const float* hb, float* hc, size_t N, size_t threadsPerBlock);

    void matrix_mul(const float* ha, const float* hb, float* hc, size_t N, size_t threadsPerBlock);
};
