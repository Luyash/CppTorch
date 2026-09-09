#include <CppTorch/Layers/Dense.h>

namespace CppTorch
{
    Dense::Dense(
        int input_size,
        int output_size,
        std::unique_ptr<Activation> activation
    )
        : activation(std::move(activation))
    {
        weights = Eigen::MatrixXd::Ones(
            input_size,
            output_size
        );

        bias = Eigen::MatrixXd::Zero(
            1,
            output_size
        );
    }

    Eigen::MatrixXd Dense::forward(const Eigen::MatrixXd& input)
    {
        last_input = input;
        last_z = input * weights + bias;
        return activation->forward(last_z);
    }
}