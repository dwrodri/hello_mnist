//
// Created by Derek Rodriguez on 12/12/2017.
//

#include "hellonet.h"


HelloNet::HelloNet(std::vector<unsigned long> layer_config): num_layers(layer_config.size()), layerConfig(layer_config){
    std::random_device rd; //apparently rand() sucks balls, so here's a Mersenne twister
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    //instantiate weight tables
    weights.resize(num_layers-1); // one table per layer (minus input layer)
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
                sourceWeight = dist(mt);
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
           neuron_bias = dist(mt);
        }
    }
}

//test constructor
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

//debug function prints all weights
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


void HelloNet::forwardProp(std::vector<float> &data) { //forward prop example
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

//this function is drenched in comments because it's so messy
void HelloNet::backProp(std::vector<float> &trainingLabel,
                        std::vector<float> &trainingData,
                        std::vector<std::vector<float>> &nablaB,
                        std::vector<std::vector<float>> &nablaW) {
    std::vector<std::vector<float>> hypotheses; //these are the "zs" from Neilsen's code
    std::vector<std::vector<float>> activations; //activation(z)
    std::vector<std::vector<float>> sp;
    activations.resize(num_layers); //pre-allocate memory
    sp.resize(num_layers - 1); //not backprop ping into output layer.
    hypotheses.resize(num_layers - 1); //no hypothesis for input layer
    activations[0] = trainingData; //the input layer just forwards the input data


    //perform forward pass
    for (int layer = 0; layer < weights.size(); ++layer) {  //for each non-input layer, get table
        activations[layer+1].resize(weights[layer].size()); //more pre-allocation for values within each layer
        hypotheses[layer].resize(weights[layer].size());
        for (int neuron = 0; neuron < weights[layer].size(); ++neuron) { //for each table, get row of weights
            float h = biases[layer][neuron];  //hypothesis h = b + ∑wa
            for (int i = 0; i < weights[layer][neuron].size(); ++i) { //compute ∑wa and add to b
                h += weights[layer][neuron][i]*activations[layer][i]; //activations[0] is just the input data
            }
            hypotheses[layer][neuron] = h;
            activations[layer+1][neuron] = activate(h); //+1 here b/c weights don't have the input layer
        }
    }

    //time to get nasty with the backward pass
    std::vector<float> delta_L;
    delta_L.resize(layerConfig.back());
    costDerivative(trainingLabel, activations.back(), delta_L); //this is the delta from the last equation
    for (int j = 0; j < layerConfig.back(); ++j) { //deltaL = (y[j]-a[j]) * actPrime(z)
        delta_L[j] *= activations.back()[j];
    }

    nablaB[nablaB.size()-1] = delta_L; //update last layer of biases
    for (int k = 0; k < nablaW.back().size(); ++k) { //generate layer of weight nablas
        nablaW.back()[k] = actPrime(activations.back()[k]) * delta_L[k];
    }

    //generate the rest of the nablas for the network for this example
    for(unsigned long l = nablaW.size()-2; l > 0; --l) {
        for (int i = 0; i < nablaW[l].size(); ++i) {
            //build vector of outputs
        }
    }




}

void HelloNet::gradientDescent(float learnRate, std::vector<std::vector<float>> &trainingData,
                               std::vector<std::vector<float>> &labels) {

    //allocate arrays for back propagation to do its thing
    std::vector<std::vector<float>> nablaB;
    std::vector<std::vector<float>> nablaW;

    nablaW.resize(num_layers-1); // one table per layer (minus input layer)
    for (int i = 1; i < num_layers; ++i) {
        nablaW[i-1].resize(layerConfig[i]); // one row per neuron in current layer
    }

    nablaB.resize(num_layers-1);
    for (int i = 1; i < num_layers; ++i) {
        nablaB[i-1].resize(layerConfig[i]);
    }

    for (int i = 0; i < labels.size(); ++i) {
        backProp(trainingData[i], labels[i], nablaB, nablaW);
    }
}

//TODO: make this multi-threaded
void HelloNet::sgd(unsigned long epochs,
                   float learnRate,
                   std::vector<std::vector<float>> &trainingData,
                   std::vector<std::vector<float>> &labels) {

    unsigned long batchSize = trainingData.size()/epochs; //size of batch
    for (unsigned long i = 0; i < (trainingData.size() - batchSize); i+= batchSize) {
        auto trainingBatch = std::vector<std::vector<float>>(trainingData.begin()+i, trainingData.end() + (i + batchSize));
        auto labelBatch = std::vector<std::vector<float>>(labels.begin() + i, labels.end() + (i + batchSize));

    }
    //parse each batch

}






