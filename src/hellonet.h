/*
 * Created by Derek Rodriguez on 12/12/2017
 * The constructor for the Neural Network.
 */

#ifndef HELLO_MNIST_HELLONET_H
#define HELLO_MNIST_HELLONET_H
#endif HELLO_MNIST_HELLONET_H
#include <cstdlib>
#include <random>
#include <vector>
#include <iostream>

class HelloNet{
private:
    //array of weighs
    std::vector<std::vector<std::vector<float>>>  weights;

    //array of bias vectors
    std::vector<std::vector<float>> biases;

    //the number of layers in the neural net
    unsigned long num_layers;

    //the amount of neurons in each layer in array form
    unsigned long  *layerData;
public:
    /*
     * HelloNet takes a list of ints, each int being the count of neurons in each layer
     * arg_count: amount of layers i.e. amount of ints in array
     * arg_list: array of ints
     */
    explicit HelloNet(std::vector<unsigned long> layer_config);

    HelloNet(std::vector<unsigned long> layer_config, float fixed_weight, float fixed_bias);

    //print the weight tables to stdout
    void dumpWeightTables();

    //activation function applied to hypothesis h
    float activate(float h);

    //derivative of activation function
    float actPrime(float h);

    //classify input data and return result into the same variable passed in
    void parse(std::vector<float> &data);

    //perform back propagation algorithm on a single training sample
    void backProp(std::vector<float> &trainingLabel, std::vector<float> &trainingData, std::vector<std::vector<float>> &nabla_b, std::vector<std::vector<float>> &nabla_w);

    //get the d/dC by performing a piecewise subtraction between two vectors and gets loaded into output á la C-style
    void costDerivative(std::vector<float> &expectedValues, std::vector<float> &currentValues, std::vector<float> &output);

    //train the NN using stochastic gradient descent
    void sgd(int epochs, float learn_rate, std::vector<std::vector<float>> &training_data, std::vector<float> &labels);


};
