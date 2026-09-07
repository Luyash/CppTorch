#pragma once

#include <Eigen/Dense>
#include "Layers.h"

namespace CppTorch
{
    class Dense : public Layer
    {
    private:
        Eigen::MatrixXd weights;
        Eigen::MatrixXd bias;

    public:
        Dense(int input_size, int output_size);

        Eigen::MatrixXd forward(
            const Eigen::MatrixXd& input
        ) override;
    };
}