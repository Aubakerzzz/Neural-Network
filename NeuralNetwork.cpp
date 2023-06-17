#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>

// Neuron class
class Neuron {
public:
    std::vector<double> weights;
    double output;
    double delta;

    Neuron(int numInputs) {
        // Initialize random weights for each input
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        for (int i = 0; i < numInputs; ++i) {
            weights.push_back(dis(gen));
        }
    }
};

// Layer class
class Layer {
public:
    int numNeurons;
    std::vector<Neuron> neurons;

    Layer(int numNeurons, int numInputsPerNeuron) {
        this->numNeurons = numNeurons;
        for (int i = 0; i < numNeurons; ++i) {
            neurons.emplace_back(Neuron(numInputsPerNeuron));
        }
    }
};

// NeuralNetwork class
class NeuralNetwork {
public:
    std::vector<Layer> layers;

    NeuralNetwork(const std::vector<int>& layerSizes) {
        for (int i = 0; i < layerSizes.size(); ++i) {
            int numNeurons = layerSizes[i];
            int numInputsPerNeuron = (i == 0) ? 1 : layerSizes[i - 1];
            layers.emplace_back(Layer(numNeurons, numInputsPerNeuron));
        }
    }

    double activationFunction(double x) {
        // Example of the sigmoid activation function
        return 1.0 / (1.0 + exp(-x));
    }

    double activationFunctionDerivative(double x) {
        // Derivative of the sigmoid activation function
        double sigmoid = activationFunction(x);
        return sigmoid * (1.0 - sigmoid);
    }

    void forwardPropagation(const std::vector<double>& inputs) {
        // Set the input values in the first layer's neurons
        for (int i = 0; i < inputs.size(); ++i) {
            layers[0].neurons[i].output = inputs[i];
        }

        // Propagate the inputs forward
        for (int i = 1; i < layers.size(); ++i) {
            Layer& prevLayer = layers[i - 1];
            Layer& currentLayer = layers[i];

            for (int j = 0; j < currentLayer.numNeurons; ++j) {
                double sum = 0.0;
                for (int k = 0; k < prevLayer.numNeurons; ++k) {
                    sum += prevLayer.neurons[k].output * currentLayer.neurons[j].weights[k];
                }
                currentLayer.neurons[j].output = activationFunction(sum);
            }
        }
    }

    void backwardPropagation(const std::vector<double>& targets, double learningRate) {
        // Calculate output layer deltas
        Layer& outputLayer = layers.back();
        for (int i = 0; i < outputLayer.numNeurons; ++i) {
            double output = outputLayer.neurons[i].output;
            double error = targets[i] - output;
            outputLayer.neurons[i].delta = error * activationFunctionDerivative(output);
        }

        // Propagate the deltas backward
        for (int i = layers.size() - 2; i >= 0; --i) {
            Layer& currentLayer = layers[i];
            Layer& nextLayer = layers[i + 1];

            for (int j = 0; j < currentLayer.numNeurons; ++j) {
                double error = 0.0;
                for (int k = 0; k < nextLayer.numNeurons; ++k) {
                    error += nextLayer.neurons[k].delta * nextLayer.neurons[k].weights[j];
                }
                currentLayer.neurons[j].delta = error * activationFunctionDerivative(currentLayer.neurons[j].output);
            }
        }

        // Update weights
        for (int i = layers.size() - 1; i > 0; --i) {
            Layer& currentLayer = layers[i];
            Layer& prevLayer = layers[i - 1];

            for (int j = 0; j < currentLayer.numNeurons; ++j) {
                for (int k = 0; k < prevLayer.numNeurons; ++k) {
                    double output = prevLayer.neurons[k].output;
                    double delta = currentLayer.neurons[j].delta;
                    currentLayer.neurons[j].weights[k] += learningRate * output * delta;
                }
            }
        }
    }

    void train(const std::vector<std::vector<double>>& trainingData,
        const std::vector<std::vector<double>>& targetData,
        int numEpochs, double learningRate) {
        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            double averageError = 0.0;
            for (int i = 0; i < trainingData.size(); ++i) {
                const std::vector<double>& inputs = trainingData[i];
                const std::vector<double>& targets = targetData[i];

                forwardPropagation(inputs);
                backwardPropagation(targets, learningRate);

                double error = 0.0;
                Layer& outputLayer = layers.back();
                for (int j = 0; j < outputLayer.numNeurons; ++j) {
                    double output = outputLayer.neurons[j].output;
                    error += 0.5 * std::pow(targets[j] - output, 2);
                }
                averageError += error;
            }
            averageError /= trainingData.size();
            std::cout << "Epoch " << epoch + 1 << ", Average Error: " << averageError << std::endl;

            // Display neuron layer values
            std::cout << "Layer Values: ";
            for (const Layer& layer : layers) {
                for (const Neuron& neuron : layer.neurons) {
                    std::cout << std::fixed << std::setprecision(5) << neuron.output << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    std::vector<double> predict(const std::vector<double>& inputs) {
        forwardPropagation(inputs);
        std::vector<double> predictions;
        Layer& outputLayer = layers.back();
        for (int i = 0; i < outputLayer.numNeurons; ++i) {
            predictions.push_back(outputLayer.neurons[i].output);
        }
        return predictions;
    }
};

int main() {
    // Define the layer sizes
    std::vector<int> layerSizes = { 2, 30, 1 }; // Example network with 2 input neurons, 3 neurons in the hidden layer, and 1 output neuron

    // Create a neural network
    NeuralNetwork neuralNetwork(layerSizes);

    // Define the training data
    std::vector<std::vector<double>> trainingData = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {0.2, 0.7},
        {0.8, 0.1},
        {0.3, 0.9},
        {0.6, 0.4},
        {0.1, 0.3},
        {0.9, 0.8},
        {0.4, 0.5},
        {0.7, 0.3},
        {0.2, 0.6},
        {0.5, 0.2},
        {0.3, 0.8},
        {0.6, 0.1},
        {0.8, 0.9},
        {0.1, 0.4},
        {0.9, 0.7},
        {0.5, 0.5},
    };

    // Define the target data
    std::vector<std::vector<double>> targetData = {
        {0.0},
        {1.0},
        {1.0},
        {0.0},
        {0.3},
        {0.7},
        {0.2},
        {0.5},
        {0.2},
        {0.9},
        {0.6},
        {0.4},
        {0.8},
        {0.3},
        {0.7},
        {0.1},
        {0.9},
        {0.4},
        {0.6},
        {0.5},
    
    };

    // Add 100 additional random training and target data points
    srand(time(0)); // Seed the random number generator

    for (int i = 0; i < 100; i++) {
        double input1 = static_cast<double>(rand()) / RAND_MAX;
        double input2 = static_cast<double>(rand()) / RAND_MAX;
        double target = static_cast<double>(rand()) / RAND_MAX;

        trainingData.push_back({ input1, input2 });
        targetData.push_back({ target });
    }

    // Train the neural network
    int numEpochs = 1000;
    double learningRate = 0.1;
    neuralNetwork.train(trainingData, targetData, numEpochs, learningRate);

    // Test the trained network
    std::vector<double> inputs = { 0.5, 0.8 }; // Example input values
    std::vector<double> predictions = neuralNetwork.predict(inputs);

    // Print the predictions
    std::cout << "Predictions: ";
    for (const double& prediction : predictions) {
        std::cout << std::fixed << std::setprecision(5) << prediction << " ";
    }
    std::cout << std::endl;

    return 0;
}
