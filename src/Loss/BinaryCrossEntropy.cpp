#include <CppTorch/Loss/BinaryCrossEntropy.h>

namespace CppTorch
{
    double CrossEntropy(
        const Eigen::MatrixXd& predicted,
        const Eigen::MatrixXd& actual
    )
    {
        double loss = 0.0;
        int n = predicted.size();
        for (int i = 0; i < n; i++)
        {
            double p = predicted(i);
            double y = actual(i);
            // Prevent log(0)
            if (p < 1e-15)
                p = 1e-15;
            if (p > 1.0 - 1e-15)
                p = 1.0 - 1e-15;

            loss += y * std::log(p) + (1.0 - y) * std::log(1.0 - p);
        }
        loss = -loss / n;
        return loss;
    }
}