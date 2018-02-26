//
// Created by Derek Rodriguez on 12/12/2017.
//

#include "hellonet.h"


HelloNet::HelloNet(std::vector<unsigned long> layer_config): num_layers(layer_config.size()){
    std::random_device rd; //apparently rand() sucks balls, so here's a Mersenne twister
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    //instantiate weight tables
    weights.resize(num_layers-1); // one table per layer
    for (int i = 1; i < num_layers; ++i) {
        weights[i-1].resize(layer_config[i]); // one row per neuron in current layer
        for (int j = 0; j < layer_config[i]; ++j) {
            weights[i-1][j].resize(layer_config[i-1]); // one entry in the row per input connection
        }
    }

    //fill weight tables with randomness
    for (auto &&layer : weights) {
        for (auto &&destination : layer) {
            for (auto &&sourceWeight : destination) {
                dist(mt);
            }
        }
    }

    //allocate space biases
    biases.resize(num_layers-1);
    for (int i = 1; i < num_layers; ++i) {
        biases[i-1].resize(layer_config[i]);
    }

    //fill bias vectors with randomness
    for (auto &&layer : biases) {
        for (auto &&neuron_bias : layer) {
            dist(mt);
        }
    }
}

//this constructor is for testing, to calculate errors in arithmetic
HelloNet::HelloNet(std::vector<unsigned long> layer_config, float fixed_weight, float fixed_bias): num_layers(layer_config.size()){
    //instantiate weight tables
    weights.resize(num_layers-1); // one table per layer
    for (int i = 1; i < num_layers; ++i) {
        weights[i-1].resize(layer_config[i]); // one row per neuron in current layer
        for (int j = 0; j < layer_config[i]; ++j) {
            weights[i-1][j].resize(layer_config[i-1]); // one entry in the row per input connection
        }
    }

    //fill weight tables with fixed value
    for (auto &&layer : weights) {
        for (auto &&destination : layer) {
            for (auto &&sourceWeight : destination) {
                sourceWeight = fixed_weight;
            }
        }
    }

    //allocate space biases
    biases.resize(num_layers-1);
    for (int i = 1; i < num_layers; ++i) {
        biases[i-1].resize(layer_config[i]);
    }

    //fill bias vectors with 0
    for (auto &&layer : biases) {
        for (auto &&neuron_bias : layer) {
            neuron_bias = fixed_bias;
        }
    }
}

void HelloNet::dumpWeightTables() {

    std::string test;
    for (auto &&layer : weights) {
        test += "NEW TABLE\n";
        for (auto &&destination : layer) {
            for (auto &&source : destination) {
                test += std::to_string(source) + "\t";
            }
            test += "\n";
        }
        test += "\nXXXXXXXXXXXXXXXXX\n";
    }

    std::cout<<test<<std::endl;

}

float HelloNet::activate(float h) { //this could be replaced with a ReLU or inverse tangent
    return 1.0f/(1.0f+expf(-h));
}

float HelloNet::actPrime(float h) { //this is just the derivative of activate(h)
    return (1.0f/(1.0f+expf(-h))) * (1.0f - 1.0f / (1.0f + expf(-h)));
}

void HelloNet::parse(std::vector<float> &data) { //forward prop example
    std::vector<float> activations; //activation
    for (int layer = 0; layer < weights.size(); ++layer) {  //for each layer, get table
        std::cout << "In layer:\t" << layer << std::endl; //DEBUG
        for (int neuron = 0; neuron < weights[layer].size(); ++neuron) { //for each table, get row of weights
            std::cout << "\tneuron:\t" << neuron; //DEBUG
            float h = biases[layer][neuron];  //hypothesis h = b + ∑wa
            for (int i = 0; i < weights[layer][neuron].size(); ++i) { //compute ∑wa and add to b
                h += weights[layer][neuron][i]*data[i];
            }
            std::cout << "\tgot a weighted avg of:\t" << h;
            activations.push_back(activate(h)); //normalize at the very end
            std::cout << "\tis outputting:\t" << activations.back() << std::endl; //DEBUG
        }
        data = activations; //activations from current layer become inputs in next layer
        activations.clear(); //clean out activations
    }
}

void HelloNet::costDerivative(std::vector<float> &expectedValues, std::vector<float> &currentValues, std::vector<float> &output) {
    std::transform(currentValues.begin(), currentValues.end(),
                   expectedValues.begin(),
                   std::back_inserter(output), std::minus<>());
}

void HelloNet::backProp(std::vector<float> &trainingLabel, std::vector<float> &trainingData, std::vector<std::vector<float>> &nabla_b, std::vector<std::vector<float>> &nabla_w) {
    std::vector<std::vector<float>> hypotheses; //these are the "zs" from Neilsen's code
    std::vector<std::vector<float>> activations;
    activations.reserve(num_layers); //pre-allocate memory for layers
    hypotheses.reserve(num_layers - 1); //no hypothesis for input layer
    activations[0] = trainingData; //the input layer just forwards the input data

    //perform forward pass
    for (int layer = 0; layer < weights.size(); ++layer) {  //for each non-input layer, get table
        activations[layer+1].reserve(weights[layer].size()); //more pre-allocation for values within each layer
        hypotheses[layer].reserve(weights[layer].size());
        for (int neuron = 0; neuron < weights[layer].size(); ++neuron) { //for each table, get row of weights
            float h = biases[layer][neuron];  //hypothesis h = b + ∑wa
            for (int i = 0; i < weights[layer][neuron].size(); ++i) { //compute ∑wa and add to b
                h += weights[layer][neuron][i]*activations[layer][i]; //activations[0] is just the input data
            }
            hypotheses[layer][neuron] = h;
            activations[layer+1][neuron] = activate(h); //+1 here b/c weights don't have the input layer
        }
    }

    //time for the backward pass
    std::vector<float> delta;
    delta.reserve(layerData[num_layers-1]); //delta vector is going to be size of the output vector
    costDerivative(trainingLabel, activations[activations.size()-1], delta);
    std::cout << "cost of example:\t<";
    for (auto &&elem : delta) {
        std::cout << elem << ", ";
    }
    std::cout << ">" << std::endl;
    for (int i  = 0; i < delta.size(); ++i){
      delta[i] *= actPrime(hypotheses[hypotheses.size()-1][i]);
    }
    nabla_b.push_back(delta); //this is the same as





}


void HelloNet::sgd(int epochs,
                   float learn_rate,
                   std::vector<std::vector<float>> &training_data,
                   std::vector<float> &labels) {

}
