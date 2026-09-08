#include "../../include/CppTorch/Activation/Linear.h"

namespace CppTorch
{
    Eigen::MatrixXd Linear::forward(
        const Eigen::MatrixXd& input
    )
    {
        return input;
    }

    Eigen::MatrixXd Linear::derivative(
        const Eigen::MatrixXd& input
    )
    {
        return Eigen::MatrixXd::Ones(
            input.rows(),
            input.cols()
        );
    }
}