#pragma once

#include <Eigen/Dense>

namespace CppTorch
{
    class Layer
    {
    public:

        virtual Eigen::MatrixXd forward(
            const Eigen::MatrixXd& input
        ) = 0;

        virtual Eigen::MatrixXd backward(
            const Eigen::MatrixXd& gradient
        ) = 0;

        virtual ~Layer() = default;
    };
}