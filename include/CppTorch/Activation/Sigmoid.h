#pragma once

#include "Activation.h"

namespace CppTorch
{
    class Sigmoid : public Activation
    {
    public:
        Eigen::MatrixXd forward(
            const Eigen::MatrixXd& input
        ) override;
    };
}