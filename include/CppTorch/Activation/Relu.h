#pragma once

#include "Activation.h"

namespace CppTorch
{
    class ReLU : public Activation
    {
    public:
        Eigen::MatrixXd forward(
            const Eigen::MatrixXd& input
        ) override;
    };
}