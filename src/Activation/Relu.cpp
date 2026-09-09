#include "../../include/CppTorch/Activation/Relu.h"

namespace CppTorch
{
    Eigen::MatrixXd ReLU::forward(
        const Eigen::MatrixXd& input
    )
    {
        return input.array().max(0.0);
    }

    Eigen::MatrixXd ReLU::derivative(
        const Eigen::MatrixXd& input
    )
    {
        return (input.array() > 0.0).cast<double>();
    }
}