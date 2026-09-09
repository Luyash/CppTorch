#include <iostream>
#include <Eigen/Dense>

#include <CppTorch/Data/Dataset.h>
#include <CppTorch/Layers/Dense.h>
#include <CppTorch/Activation/ReLU.h>
#include <CppTorch/Activation/Sigmoid.h>
#include "../include/CppTorch/Loss/BinaryCrossEntropy.h"


int main()
{
    // ============================================================
    // 1. TRAINING DATA
    // ============================================================
    //
    // Feature 1 = Hours studied
    // Feature 2 = Sickness level
    //
    // Sickness:
    // 0 = completely healthy
    // 1 = very sick
    //
    // Target:
    // 0 = Fail
    // 1 = Pass
    //
    // ------------------------------------------------------------

    Eigen::MatrixXd X(10, 2);

    X <<

        2, 0.9,
        3, 0.8,
        4, 0.7,
        5, 0.6,
        6, 0.4,
        7, 0.3,
        8, 0.2,
        9, 0.1,
        10, 0.0,
        1, 1.0;


    Eigen::MatrixXd y(10, 1);

    y <<
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0;


    CppTorch::Dataset data(X, y);


    // ============================================================
    // 2. CREATE NEURAL NETWORK
    // ============================================================

    CppTorch::Dense layer1(
        2,
        20,
        std::make_unique<CppTorch::ReLU>()
    );

    CppTorch::Dense layer2(
        20,
        10,
        std::make_unique<CppTorch::ReLU>()
    );

    CppTorch::Dense output(
        10,
        1,
        std::make_unique<CppTorch::Sigmoid>()
    );


    // ============================================================
    // 3. TRAINING SETTINGS
    // ============================================================

    const int epochs = 10000;

    const double learning_rate = 0.01;


    // ============================================================
    // 4. TRAINING LOOP
    // ============================================================

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        // --------------------------------------------------------
        // FORWARD PASS
        // --------------------------------------------------------

        Eigen::MatrixXd a1 =
            layer1.forward(data.getInputs());

        Eigen::MatrixXd a2 =
            layer2.forward(a1);

        Eigen::MatrixXd prediction =
            output.forward(a2);


        // --------------------------------------------------------
        // LOSS
        // --------------------------------------------------------

        double loss =
            CppTorch::CrossEntropy(
                prediction,
                data.getTargets()
            );


        // --------------------------------------------------------
        // BACKPROPAGATION
        // --------------------------------------------------------

        //
        // For BCE + Sigmoid:
        //
        // dL/dz = prediction - actual
        //
        // This is a useful simplification.
        //

        Eigen::MatrixXd gradient =
            prediction - data.getTargets();


        gradient =
            output.backward(gradient);

        gradient =
            layer2.backward(gradient);

        gradient =
            layer1.backward(gradient);


        // --------------------------------------------------------
        // GRADIENT DESCENT
        // --------------------------------------------------------

        //
        // Your Dense class should use its calculated gradients
        // to update:
        //
        // weights -= learning_rate * dW
        // bias    -= learning_rate * dB
        //
        // --------------------------------------------------------


        if (epoch % 1000 == 0)
        {
            std::cout
                << "Epoch: "
                << epoch
                << " | Loss: "
                << loss
                << std::endl;
        }
    }


    // ============================================================
    // 5. USER INTERFACE
    // ============================================================

    std::cout << "\n============================\n";
    std::cout << "   PASS / FAIL PREDICTOR\n";
    std::cout << "============================\n";


    double hours;
    double sickness;

    std::cout << "\nEnter hours studied: ";
    std::cin >> hours;

    std::cout << "Enter sickness level (0 - 1): ";
    std::cin >> sickness;


    // Create input matrix

    Eigen::MatrixXd userInput(1, 2);

    userInput << hours, sickness;


    // ============================================================
    // 6. PREDICTION
    // ============================================================

    Eigen::MatrixXd userA1 =
        layer1.forward(userInput);

    Eigen::MatrixXd userA2 =
        layer2.forward(userA1);

    Eigen::MatrixXd userPrediction =
        output.forward(userA2);


    double probability =
        userPrediction(0, 0);


    // ============================================================
    // 7. PASS / FAIL
    // ============================================================

    std::cout << "\nProbability of passing: "
              << probability * 100
              << "%\n";


    if (probability >= 0.5)
    {
        std::cout << "Prediction: PASS\n";
    }
    else
    {
        std::cout << "Prediction: FAIL\n";
    }


    return 0;
}