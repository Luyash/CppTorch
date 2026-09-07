#pragma once

#include <Eigen/Dense>

namespace CppTorch
{
    double MSE(
        const Eigen::MatrixXd& predicted,
        const Eigen::MatrixXd& actual
    );
}