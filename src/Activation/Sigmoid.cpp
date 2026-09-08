#include "../../include/CppTorch/Activation/Sigmoid.h"

namespace CppTorch
{
    Eigen::MatrixXd Sigmoid::forward(
        const Eigen::MatrixXd& input
    )
    {
        return 1.0 / (1.0 + (-input.array()).exp());
    }
}