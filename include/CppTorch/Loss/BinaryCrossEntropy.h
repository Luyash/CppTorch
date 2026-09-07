#pragma once

#include <Eigen/Dense>

namespace CppTorch
{
    double CrossEntropy(
        const Eigen::MatrixXd& predicted,
        const Eigen::MatrixXd& actual
    );
}