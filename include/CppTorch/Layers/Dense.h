#pragma once

#include <Eigen/Dense>
#include <memory>

#include "Layers.h"
#include "../Activation/Activation.h"

namespace CppTorch
{
    class Dense : public Layer
    {
    private:
        Eigen::MatrixXd weights;
        Eigen::MatrixXd bias;

        std::unique_ptr<Activation> activation;

    public:
        Dense(
            int input_size,
            int output_size,
            std::unique_ptr<Activation> activation
        );

        Eigen::MatrixXd forward(
            const Eigen::MatrixXd& input
        ) override;
    };
}