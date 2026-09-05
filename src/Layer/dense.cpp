#include "include/Layers/Dense.h"

namespace CppTorch
{
    Dense::Dense(int input_size, int output_size)
    {
        weights = Eigen::MatrixXd::Ones(input_size, output_size);
        bias = Eigen::MatrixXd::Zero(1, output_size);
    }

    Eigen::MatrixXd Dense::forward(
        const Eigen::MatrixXd& input
    )
    {
        return input * weights + bias;
    }
}