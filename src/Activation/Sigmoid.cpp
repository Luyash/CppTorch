#include "../../include/CppTorch/Activation/Sigmoid.h"

namespace CppTorch
{
    Eigen::MatrixXd Sigmoid::forward(
        const Eigen::MatrixXd& input
    )
    {
        return 1.0 / (1.0 + (-input.array()).exp());
    }

    Eigen::MatrixXd Sigmoid::derivative(
        const Eigen::MatrixXd& input
    )
    {
        Eigen::MatrixXd output = forward(input);

        return (output.array() * (1.0 - output.array())).matrix();
    }
}