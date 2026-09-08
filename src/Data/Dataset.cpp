#include "../../include/CppTorch/Data/Dataset.h"

namespace CppTorch
{
    Dataset::Dataset(
        const Eigen::MatrixXd& inputs,
        const Eigen::MatrixXd& targets
    )
    {
        this->inputs = inputs;
        this->targets = targets;
    }

    Eigen::MatrixXd Dataset::getInputs() const
    {
        return inputs;
    }

    Eigen::MatrixXd Dataset::getTargets() const
    {
        return targets;
    }
}