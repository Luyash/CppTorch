#include <iostream>

#include "include/Layers/Dense.h"

int main()
{
    // Create a Dense layer
    CppTorch::Dense layer(2, 1);

    // One input with 2 features
    Eigen::MatrixXd input(1, 2);

    input << 3, 4;

    // Forward propagation
    Eigen::MatrixXd output = layer.forward(input);

    std::cout << "Output:\n";
    std::cout << output << std::endl;

    return 0;
}