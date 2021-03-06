//
// Created by Derek Rodriguez on 12/12/2017.
//

#include "hellonet.h"


HelloNet::HelloNet(std::vector<unsigned long> layer_config): num_layers(layer_config.size()), layerConfig(layer_config){
    std::random_device rd; //apparently rand() sucks balls, so here's a Mersenne twister
    std::mt19937 mt(rd());
    std::uniform_real_distribution<long double> dist(-1.0, 1.0);

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
HelloNet::HelloNet(std::vector<unsigned long> layer_config, long double fixed_weight, long double fixed_bias): num_layers(layer_config.size()){
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


long double HelloNet::activate(long double h) { //this could be replaced with a ReLU or inverse tangent
    return 1.0f/(1.0f+expf(-h));
}


long double HelloNet::actPrime(long double h) { //this is just the derivative of activate(h)
    return (1.0f/(1.0f+expf(-h))) * (1.0f - 1.0f / (1.0f + expf(-h)));
}


void HelloNet::forwardProp(std::vector<long double> &data) { //forward prop example
    std::vector<long double> activations; //activation
    for (int layer = 0; layer < weights.size(); ++layer) {  //for each layer, get table
        std::cout << "In layer:\t" << layer << std::endl; //DEBUG
        for (int neuron = 0; neuron < weights[layer].size(); ++neuron) { //for each table, get row of weights
            std::cout << "\tneuron:\t" << neuron; //DEBUG
            long double h = biases[layer][neuron];  //hypothesis h = b + ∑wa
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


void HelloNet::costDerivative(std::vector<long double> &expectedValues, std::vector<long double> &currentValues, std::vector<long double> &output) {
    std::transform(currentValues.begin(), currentValues.end(),
                   expectedValues.begin(),
                   std::back_inserter(output), std::minus<long double>());
}

//this function is drenched in comments because it's so messy
void HelloNet::backProp(std::vector<long double> &trainingLabel,
                        std::vector<long double> &trainingData,
                        std::vector<std::vector<long double> > &nablaB,
                        std::vector<std::vector<std::vector<long double> > > &nablaW) {
    std::vector<std::vector<long double> > hypotheses; //these are the "zs" from Neilsen's code
    std::vector<std::vector<long double> > activations; //activation(z)
    std::vector<std::vector<long double> > sp;
    activations.resize(num_layers); //pre-allocate memory
    sp.resize(num_layers - 1); //not backprop ping into output layer.
    hypotheses.resize(num_layers - 1); //no hypothesis for input layer
    activations[0] = trainingData; //the input layer just forwards the input data


    //perform forward pass
    for (int layer = 0; layer < weights.size(); ++layer) {  //for each non-input layer, get table
        activations[layer+1].resize(weights[layer].size()); //more pre-allocation for values within each layer
        hypotheses[layer].resize(weights[layer].size());
        for (int neuron = 0; neuron < weights[layer].size(); ++neuron) { //for each table, get row of weights
            long double h = biases[layer][neuron];  //hypothesis h = b + ∑wa
            for (int i = 0; i < weights[layer][neuron].size(); ++i) { //compute ∑wa and add to b
                h += weights[layer][neuron][i]*activations[layer][i]; //activations[0] is just the input data
            }
            hypotheses[layer][neuron] = h;
            activations[layer+1][neuron] = activate(h); //+1 here b/c weights don't have the input layer
        }
    }

    //time to get nasty with the backward pass
    std::vector<long double> delta_L;
    delta_L.resize(layerConfig.back());
    costDerivative(trainingLabel, activations.back(), delta_L); //this is the delta from the last equation
    for (int j = 0; j < layerConfig.back(); ++j) { //deltaL = (y[j]-a[j]) * actPrime(z)
        delta_L[j] *= activations.back()[j];
    }

    nablaB[nablaB.size()-1] = delta_L; //update last layer of biases
    for (int k = 0; k < nablaW.back().size(); ++k) { //generate layer of weight nablas
        for (int i = 0; i < nablaW.back()[k].size(); ++i) {
            nablaW.back()[k][i] = activations[activations.size()-1][i]*delta_L[k];
        }
    }

    //generate the rest of the nablas for the network for this example
    for(unsigned long l = nablaW.size()-2; l > 1; --l) { //for each layer, moving backwards
        //build transpose of weight table
        std::vector<std::vector<long double> > transposedWeightTable;
        //pre-allocate memory for transpose
        transposedWeightTable.resize(weights[l+1][0].size());
        for (int i = 0; i < weights[l+1][0].size(); ++i) {
            transposedWeightTable[i].resize(weights[l+1].size());
        }

        for (int j = 0; j < weights[l + 1].size(); ++j) {
            for (int i = 0; i < weights[l + 1][j].size(); ++i) {
                transposedWeightTable[i][j] = weights[l+1][j][i]; //load in crisscrossed for transpose
            }
        }

        for (int k = 0; k < transposedWeightTable.size(); ++k) { //
            nablaB[l][k] = std::inner_product(transposedWeightTable[k].begin(), transposedWeightTable[k].end(), delta_L.begin(), 0.0f);
            nablaB[l][k] *= actPrime(hypotheses[l][k]); //this is where you compute the layer delta vector
            for (int i = 0; i < nablaW[l][k].size(); ++i) {
                nablaW[l][k][i] = activations[l-1][i]*nablaB[l][k];
            }
        }
        delta_L = nablaB[l];

    }




}

void HelloNet::gradientDescend(long double learnRate, std::vector<std::vector<long double> > &trainingData,
                               std::vector<std::vector<long double> > &labels) {

    //allocate arrays for back propagation to do its thing
    std::vector<std::vector<long double> > nablaB;
    std::vector<std::vector<std::vector<long double> > > nablaW;

    nablaW.resize(num_layers-1); // one table per layer (minus input layer)
    for (int i = 1; i < num_layers; ++i) {
        nablaW[i-1].resize(layerConfig[i]); // one row per neuron in current layer
        for (int j = 0; j < layerConfig[i]; ++j) {
            nablaW[i-1][j].resize(layerConfig[i-1]); // one entry in the row per input connection
        }
    }

    nablaB.resize(num_layers-1);
    for (int i = 1; i < num_layers; ++i) {
        nablaB[i-1].resize(layerConfig[i]);
    }

    for (int i = 0; i < labels.size(); ++i) {
        backProp(trainingData[i], labels[i], nablaB, nablaW);
    }

    //scale updates learn rate
    for (auto &&layer : nablaB) {
        for (auto &&neuronBiasNabla : layer) {
            neuronBiasNabla *= learnRate;
        }
    }
    for (auto &&layer : nablaW) {
        for (auto &&neuron : layer) {
            for (auto &&neuronWeightNabla : neuron) {
                neuronWeightNabla *= learnRate;
            }
        }
    }

    //apply updates to neural net
    for (int k = 1; k < num_layers-1; ++k) {
        for (int i = 0; i < layerConfig[k]; ++i) {
            biases[k-1][i] += nablaB[k-1][i];
            for (int j = 0; j < weights[k-1][i].size(); ++j) {
                weights[k-1][i][j] += nablaW[k-1][i][j]; //adjust for lack of input layer in weight tables
            }
        }
    }


}

//TODO: make this multi-threaded
void HelloNet::sgd(unsigned long epochs,
                   long double learnRate,
                   std::vector<std::vector<long double> > &trainingData,
                   std::vector<std::vector<long double> > &labels) {

    unsigned long batchSize = trainingData.size()/epochs; //size of batch
    for (unsigned long i = 0; i < (trainingData.size() - batchSize); i+= batchSize) {
        auto trainingBatch = std::vector<std::vector<long double> >(trainingData.begin()+i, trainingData.end() + (i + batchSize));
        auto labelBatch = std::vector<std::vector<long double> >(labels.begin() + i, labels.end() + (i + batchSize));

    }

}






