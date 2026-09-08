#pragma once

#include "Activation.h"

namespace CppTorch
{
    class Linear : public Activation
    {
    public:
        Eigen::MatrixXd forward(
            const Eigen::MatrixXd& input
        ) override;
    };
}