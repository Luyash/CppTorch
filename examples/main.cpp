#include <iostream>
#include <Eigen/Dense>

int main()
{
    Eigen::MatrixXd A(2,2);

    A << 1, 2,
         3, 4;

    Eigen::MatrixXd B(2,2);

    B << 5, 6,
         7, 8;

    std::cout << "Matrix A\n" << A << "\n\n";
    std::cout << "Matrix B\n" << B << "\n\n";

    std::cout << "A + B\n" << A + B << "\n\n";
    std::cout << "A * B\n" << A * B << "\n";

    std::cout<<"Juts for test";

    return 0;
}