
#include "mnist_parser.h"

MNISTDataset::MNISTDataset(std::string image_path, std::string label_path) : image_arr(), label_arr(), num_images(), rows(), cols()
{
    std::ifstream image_file(image_path.c_str(), std::ifstream::binary);
    if (image_file.is_open())
    {
        int magic_number = 0;
        image_file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = FlipEndian(magic_number);

        if(magic_number != 2051) {
            std::cout << "Invalid image file" << std::endl;
            return;
        }

        image_file.read((char*)&num_images,sizeof(num_images));
        num_images = FlipEndian(num_images);
        image_file.read((char*)&rows,sizeof(rows));
        rows= FlipEndian(rows);
        image_file.read((char*)&cols,sizeof(cols));
        cols= FlipEndian(cols);

        image_arr.resize(num_images, std::vector<double>(rows * cols));

        for(int i = 0; i < num_images; ++i)
        {
            for(int r = 0; r < rows; ++r)
            {
                for(int c = 0; c < cols; ++c)
                {
                    unsigned char temp = 0;
                    image_file.read((char*)&temp, sizeof(temp));
                    image_arr[i][(rows * r) + c] = (double)temp;
                }
            }
        }
    }
    else
    {
        std::cout << "Error opening image file" << std::endl;
    }

    std::ifstream label_file(label_path.c_str(), std::ifstream::binary);
    if (label_file.is_open())
    {
        int magic_number = 0;
        int num_labels = 0;
        label_file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = FlipEndian(magic_number);
        label_file.read((char*)&num_labels,sizeof(num_labels));
        num_labels = FlipEndian(num_labels);

        if(magic_number != 2049) {
            std::cout << "Invalid label file" << std::endl;
            return;
        }
        if(num_images != num_labels) {
            std::cout << "Label file does not match image file" << std::endl;
            return;
        }

        label_arr.resize(num_images);

        for(int i = 0; i < num_images; ++i)
        {
            unsigned char temp = 0;
            label_file.read((char*)&temp, sizeof(temp));
            label_arr[i] = (int)temp;
        }
    }
    else
    {
        std::cout << "Error opening label file" << std::endl;
    }
}
