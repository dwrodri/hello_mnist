/*
 * This is the class designed to parse the MNIST dataset. In the perfect world, all you'll have to do is write a parser
 * for your dataset, and this NN will be able to train on it. However, that might take a while.
 */

#ifndef HELLO_MNIST_MNIST_PARSER_H
#define HELLO_MNIST_MNIST_PARSER_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class MNISTDataset {
public:
    MNISTDataset(std::string image_path, std::string label_path);
    int rows;
    int cols;
    int num_images;
    std::vector< std::vector<double> > image_arr; // Could be floats, but double for calculations
    std::vector<int> label_arr;
};

static int FlipEndian(int i)
{
    int byte_1, byte_2, byte_3, byte_4;
    byte_1 = i & 255;
    byte_2 = (i >> 8) & 255;
    byte_3 = (i >> 16) & 255;
    byte_4 = (i >> 24) & 255;
    return (byte_1 << 24) + (byte_2 << 16) + (byte_3 << 8) + byte_4;
}

#endif //HELLO_MNIST_MNIST_PARSER_H
