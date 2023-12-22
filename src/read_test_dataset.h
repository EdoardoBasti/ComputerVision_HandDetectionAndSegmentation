//author: Edoardo Bastianello, ID:2053077

//these function are useful to read the test images/bounding boxes/masks

#ifndef READ_TEST_IMAGES_H
#define READ_TEST_IMAGES_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "Bounding_Box.h"


//function that reads the images in the selected directory
//INPUT PARAMETERS:	-path to the directory containing the images (in the format "rgb/", "mask/", ...)
//					-format of the images (in the format "png, "jpg", ... ---> with no dot (".") in the string)
//the test images are in "jpg" format, the test masks are in "png" format
//OUTPUT: a vector of images (images or masks)
std::vector<cv::Mat> read_test_images(const std::string& path_to_directory, const std::string format);


//function that reads the bounding boxes of an image
//INPUT PARAMETERS: path to the file (example: "det/01.txt")
//OUTPUT: a vector containing all the bounding boxes of the image
std::vector<Bounding_Box> read_Bounding_Boxes_one_img(const std::string& path_to_file_boxes_img);


//function that reads the txt files of bounding boxes in the selected directory
//INPUT PARAMETERS:	path to the directory containing the txt files (in the format "det/", ...)
//the files are in "txt" format
//OUTPUT: a vector of vectors of Bounding_Box (a vector of Bounding_Box for each image)
std::vector<std::vector<Bounding_Box>> read_test_b_box(const std::string& path_to_directory);


#endif