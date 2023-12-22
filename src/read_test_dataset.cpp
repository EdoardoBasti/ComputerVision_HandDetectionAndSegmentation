//author: Edoardo Bastianello, ID:2053077

//these function are useful to read the test images/bounding boxes/masks

#include <string>
#include <vector>
#include <stdexcept>

#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


#include "Bounding_Box.h"
#include "read_test_dataset.h"


//function that reads the images in the selected directory
//INPUT PARAMETERS:	-path to the directory containing the images (in the format "rgb/", "mask/", ...)
//					-format of the images (in the format "png, "jpg", ... ---> with no dot (".") in the string)
//the test images are in "jpg" format, the test masks are in "png" format
//OUTPUT: a vector of images (images or masks)
std::vector<cv::Mat> read_test_images(const std::string& path_to_directory, const std::string format)
{  
    //images/masks
    std::vector<cv::Mat> images;
    std::vector<std::string> file_names;
    std::string path_to_files = path_to_directory;
    path_to_files.append("*.");
    path_to_files.append(format);

    //get file names
    cv::glob(path_to_files, file_names);

    //read images/masks
    for (int i = 0; i < file_names.size(); i++)
    {
        cv::Mat img = cv::imread(file_names[i]);
        //check if image is found
        if (img.empty())
        {
            throw std::invalid_argument("Could not open or find the image\n");
        }
        //add current image/mask to the vector of all images/masks
        images.push_back(img);
    }

    return images;
}


//function that reads the bounding boxes of an image
//INPUT PARAMETERS: path to the file (example: "det/01.txt")
//OUTPUT: a vector containing all the bounding boxes of the image
std::vector<Bounding_Box> read_Bounding_Boxes_one_img(const std::string& path_to_file_boxes_img)
{
    //parsing of file txt
    std::fstream file_stream;

    //bounding boxes to be returned
    std::vector<Bounding_Box> b_boxes;

    //open file
    file_stream.open(path_to_file_boxes_img);

    //check if file is found
    if (file_stream.fail())
    {
        throw std::invalid_argument("Could not open or find the txt file with the bounding boxes\n");
    }

    //bounding box parameters
    std::string x, y, width, height;
    int x_in, y_in, width_in, height_in;

    //read file
    while (file_stream >> x >> y >> width >> height)
    {
        x_in = stoi(x);
        y_in = stoi(y);
        width_in = stoi(width);
        height_in = stoi(height);

        //add bounding box to the vector of bounding boxes of this image
        b_boxes.push_back(Bounding_Box(x_in, y_in, width_in, height_in));
    }

    return b_boxes;
}


//function that reads the txt files of bounding boxes in the selected directory
//INPUT PARAMETERS:	path to the directory containing the txt files (in the format "det/", ...)
//the files are in "txt" format
//OUTPUT: a vector of vectors of Bounding_Box (a vector of Bounding_Box for each image)
std::vector<std::vector<Bounding_Box>> read_test_b_box(const std::string& path_to_directory)
{
    //vector containing ALL the bounding boxes of ALL images
    std::vector<std::vector<Bounding_Box>> bounding_boxes_imgs;
    std::string path_to_files = path_to_directory;
    std::vector<std::string> file_names;
    path_to_files.append("*.txt");

    //find names of files
    cv::glob(path_to_files, file_names);
    for (int i = 0; i < file_names.size(); i++)
    {
        //bounding boxes of one img
        std::vector<Bounding_Box> bounding_boxes_single_img= read_Bounding_Boxes_one_img(file_names[i]);

        //bounding boxes of all imgs
        bounding_boxes_imgs.push_back(bounding_boxes_single_img);
    }

    return bounding_boxes_imgs;
}


