//Author: Gabriele Boscarini, ID 2063145

#ifndef MASK_GENERATOR_H
#define MASK_GENERATOR_H

#include <vector>
#include <string>

#include <opencv2/core.hpp>


#include "Bounding_Box.h"



//function that overlap masks in regions of interest of an image
//INPUT PARAMETERS:	-the image before paste masks (passed by const reference)
//					-the image in which masks are pasted 
//                  -the bounding boxes in wich operate the overlap
//OUTPUT: it overlaps an image, no value is returned
void overlap_image(const cv::Mat& input, cv::Mat& coloredImage, std::vector<Bounding_Box> boxes);



//function that return two possible final masks: binary or colored
//INPUT PARAMETERS:	-the initial image
//                  -vector containing masks images as result of segmentation
//                  -the bounding boxes in wich segmentation was made
//                  - a string for choice wich mask is the output: "binary" for binary mask or "colored" for colored mask
//OUTPUT: return the final mask

cv::Mat getFinalMask(const cv::Mat& in, std::vector<cv::Mat>& masks, std::vector<Bounding_Box> boxes, std::string MaskType);


#endif
