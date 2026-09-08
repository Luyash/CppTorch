#pragma once

#include <Eigen/Dense>

namespace CppTorch
{
    class Dataset
    {
    private:
        Eigen::MatrixXd inputs;
        Eigen::MatrixXd targets;

    public:
        Dataset(
            const Eigen::MatrixXd& inputs,
            const Eigen::MatrixXd& targets
        );

        Eigen::MatrixXd getInputs() const;
        Eigen::MatrixXd getTargets() const;
    };
}