#include "hellonet.h"
#include <thread>

/*
 * At this point in time, this is just a playground file where I goof around and see what works
 * Created by Derek Rodriguez on 12/10/2017
 * Last edited by: Derek Rodriguez
 */


//thread-specific function that loads random long doubles into array
void threadDataCallBack(long double* const & arr,
                        unsigned long long const & dataRange){
    std::random_device rd;
    std::mt19937 mt(rd()); //Mersenne twister could be swapped with XOR128+ algorithm
    std::uniform_real_distribution<long double> dist(-1.6f, 1.6f);

    auto localPointer = const_cast<long double* &>(arr);
    for (unsigned long long i = 0; i < dataRange; ++i) {
        localPointer[i] = dist(mt);
    }

}

//thread-specific function takes linear array of components and creates labels for unit vs. non-unit
void threadLabelCallBack(long double* const & dataArr,
                         long double* const & labelArr,
                         unsigned long long const & dataRange,
                         unsigned long long const & labelRange){
    unsigned long vecLength = dataRange / labelRange; //calc length of vector

    auto localPointer = const_cast<long double* &>(labelArr); // no need for other casts because this is the only one changed

    //there's got to be a more efficient way but I can't think of a better dynamic solution
    int labelIndex = 0;
    for (int i = 0; i < dataRange; i+=vecLength) {
        long double sqSum = 0;
        for (int j = 0; j < vecLength; ++j) {
            sqSum += dataArr[i+j] * dataArr[i+j];
        }
        localPointer[labelIndex++] = sqSum > 1 ? 1: 0;
    }
}

void testConstructorAndForwardProp(){
    //construct network
    std::cout << "Creating NN with spec: [5,4,3,2,1]" << std::endl;
    std::vector<unsigned long> config = {5, 4, 3, 2, 1}; //two inputs, two hidden, and one output for testing
    auto *test = new HelloNet(config);
    std::cout << "Here are the weight tables:" << std::endl;
    test->dumpWeightTables(); //dump weight matrix for debugging


    //forward propagation test
    std::cout << "propagating through = < 1, 1, 1, 1, 1 >" << std::endl;
    std::vector<long double> sample = {1, 1, 1, 1, 1}; //dummy data
    test->forwardProp(sample);
    std::string fPropOutput = "<";
    for (auto &&result : sample) fPropOutput += std::to_string(result) + ", "; //concatenate output to line
    fPropOutput += ">";
    std::cout << "RESULT:\t" + fPropOutput << std::endl; // as long as it's not zero we're good.

}

//TODO: Fix this code to work for data sets larger than 2^17
void testUnitCircleSeparation(long long trainingSetSize, long long testSetSize) { //separate unit vectors from non-unit vectors
    //instantiate simple network
    std::vector<unsigned long> config = {2, 2, 1}; //hard-coded configuration for then network

    //generate random numbers in parallel
    std::cout << "generating random numbers..." << std::endl;
    unsigned long numThreads = std::thread::hardware_concurrency();
    unsigned long long perThreadDataRange = (config[0]*trainingSetSize)/numThreads;
    auto sharedDataContainer = new long double[config[0] * trainingSetSize]; //data is size of input
    std::thread dataThreadList[numThreads];
    for (int i = 0; i < numThreads; ++i) {
        //NOTE: THIS IS REALLY TERRIBLE CODE, IT BREAKS WHEN TRAINING SET > 2^18
        dataThreadList[i] = std::thread(threadDataCallBack, (sharedDataContainer + i * perThreadDataRange), perThreadDataRange);
    }
    for (auto &thread : dataThreadList) {
        if(thread.joinable()){
            thread.join();
        }
    }

    std::cout << "Generating labels..." << std::endl;
    //generate labels
    auto sharedLabelContainer = new long double[trainingSetSize]; //size of output (these will always be one dimension for this problem)
    unsigned long long perThreadLabelRange = trainingSetSize / numThreads;
    std::thread labelThreadList[numThreads];

    for (int j = 0; j < numThreads; ++j) {
       labelThreadList[j] = std::thread(threadLabelCallBack,
                                   (sharedDataContainer + j * perThreadDataRange),
                                   (sharedLabelContainer + j * perThreadLabelRange),
                                   perThreadDataRange,
                                   perThreadLabelRange);
    }

    for (auto &thread : labelThreadList) {
        if(thread.joinable()){
            thread.join();
        }
    }

    //move the data and labels to STL vectors
    std::cout << "reallocating into vectors" << std::endl;
    unsigned long vecLength = perThreadDataRange / perThreadLabelRange;
    std::vector<std::vector<long double>> labels;
    std::vector<std::vector<long double>> data;
    long double percentOfSmallVectors = 0.0;
    labels.resize(static_cast<unsigned long>(trainingSetSize));
    data.resize(static_cast<unsigned long>(trainingSetSize));
    for (auto &datum : data) {
        datum.resize(vecLength);
    }
    for (int k = 0; k < trainingSetSize; ++k) { //convert 1D array into vector of n-dimensional data for input
        for (int i = 0; i < vecLength; ++i) {
            data[k][i] = sharedDataContainer[k+i];
        }
    }

    for (int l = 0; l < labels.capacity(); ++l) { //Yes, the labels are inside a vector of length 1, I know it's dumb
        labels[l] = {sharedLabelContainer[l]};
    }

    for (auto &&label : labels) {
        if (label[0] == 0) percentOfSmallVectors++;
    }

    percentOfSmallVectors /= labels.size();

    std::cout << (percentOfSmallVectors*100) << "% of vectors in this set are smaller than a unit vectors" << std::endl;


    //instantiate the network and train via gradient descent
    std::cout << "Instantiating network and passing data" << std::endl;
    auto neuralNet = new HelloNet(config);
    neuralNet->dumpWeightTables(); // for debugging purposes
    neuralNet->gradientDescend(0.5, data, labels);
    //TODO: write parsing function for testing set

}

int main(int argc, char **argv) {
    testConstructorAndForwardProp();
    testUnitCircleSeparation(atoll(argv[1]), atoll(argv[2]));
}
