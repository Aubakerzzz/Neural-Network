# Neural-Network

This is an example program that demonstrates how to create and train a neural network using the C++ programming language. The program uses a feedforward neural network with backpropagation for training.

## Getting Started

### Prerequisites

To compile and run the program, you need:

- C++ compiler (supporting C++11 or higher)
- Standard library headers: `<iostream>`, `<iomanip>`, `<cmath>`, `<vector>`, `<random>`

### Compilation

You can compile the program using the following command:
g++ NeuralNetwork.cpp -o neural_network


### Usage

After compiling the program, you can run it using the following command:

./neural_network


## Program Description

The program defines several classes:

- `Neuron`: Represents a single neuron in the neural network.
- `Layer`: Represents a layer of neurons in the neural network.
- `NeuralNetwork`: Represents the neural network itself and provides methods for training and prediction.

The main function creates an instance of the `NeuralNetwork` class, defines the training data and target data, trains the network, and tests it with example input values.

## Customization

You can customize the neural network by modifying the following variables in the `main` function:

- `layerSizes`: Specify the sizes of the network layers. For example, `{2, 30, 1}` represents a network with 2 input neurons, 30 neurons in the hidden layer, and 1 output neuron.
- `trainingData`: Define the training data as a vector of input vectors. Each input vector represents one training example.
- `targetData`: Define the corresponding target data as a vector of target vectors. Each target vector represents the desired output for the corresponding input example.
- `numEpochs`: Set the number of training epochs.
- `learningRate`: Set the learning rate for the backpropagation algorithm.

## License

This program is licensed under the MIT License. You can find the license information in the [LICENSE](LICENSE) file.

## Acknowledgments

This program is based on the concepts of neural networks and backpropagation. Special thanks to the open-source community for providing valuable resources and examples.



