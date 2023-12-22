//author: Edoardo Bastianello, ID:2053077

//file that contains the functions to:
//visualize an image 
//visualize an image with its bounding boxes
//color the masks of an image, each one with a different color

#ifndef VISUALIZE_RESULTS_H
#define VISUALIZE_RESULTS_H

#include <vector>
#include <string>

#include <opencv2/core.hpp>

#include "Bounding_Box.h"


//function that visualizes an image
//INPUT PARAMETERS:	-the image to be visualized (passed by const reference)
//					-the name of the window (passed by const reference)
//OUTPUT: it visualizes an image, no value is returned
void visualize_image(const cv::Mat& img, const std::string& window_name);


//function that adds the bounding box passed as parameter to the image passed as parameter
//INPUT PARAMETERS:	-the image to which you want to add the bounding box (passed by reference)
//					-the bounding box to be added (passed by const reference)
//					-the color of the Bounding Box (one different color for each hand in the image)
//OUTPUT: None
void draw_one_bounding_box(cv::Mat& img, const Bounding_Box& box, cv::Scalar color);


//function that visualizes an image with its bounding boxes
//INPUT PARAMETERS:	-the image to be visualized (passed by const reference)
//					-an std::vector of Bounding_Box objects (passed by const reference)
//OUTPUT: it visualizes an image, no value is returned
void visualize_results_bounding_box(const cv::Mat& img, const std::vector<Bounding_Box>& boxes);


//function that colors each mask in the vector with a different color
//INPUT PARAMETERS: a vector of masks (passed by reference)
//OUTPUT: None, it changes the colors
void color_masks(std::vector<cv::Mat>& img_masks);


#endif