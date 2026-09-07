#include <CppTorch/Loss/MSE.h>

namespace CppTorch
{
    double MSE(
        const Eigen::MatrixXd& predicted,
        const Eigen::MatrixXd& actual
    )
    {
        Eigen::MatrixXd difference = predicted - actual;
        Eigen::MatrixXd squared = difference.array().square();
        double loss = squared.mean();
        return loss;
    }
}