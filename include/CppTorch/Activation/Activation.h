#pragma once

#include <Eigen/Dense>

namespace CppTorch
{
    class Activation
    {
    public:

        virtual Eigen::MatrixXd forward(
            const Eigen::MatrixXd& input
        ) = 0;

        virtual Eigen::MatrixXd derivative(
            const Eigen::MatrixXd& input
        ) = 0;

        virtual ~Activation() = default;
    };
}