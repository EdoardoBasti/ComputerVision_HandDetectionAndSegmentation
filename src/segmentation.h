//author: Edoardo Bastianello, ID:2053077

//file to segment an image on a custom approach based on watershed intersected with a binary image (and other methods if the results obtained with watershed are not good enough)

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <vector>

#include <opencv2/core.hpp>

#include "Bounding_Box.h"


//function that counts how many white pixels there are in an image
//INPUT PARAMETERS: the image (passed by const reference)
//OUTPUT: an integer with the number of white pixels in the image
int count_white_pixels_img(const cv::Mat& img);


//function that, given an image and one of its bounding boxes, returns a cropped image with the region contained in the bounding box
//INPUT PARAMETERS: - the image (passed by const reference)
//                  - one of its bounding boxes (passed by const reference)
//OUTPUT: a cropped image containing the region of the image inside the bounding box
cv::Mat crop_Bounding_Box(const cv::Mat& img, const Bounding_Box& bbox);


//function that computes the segmentation of one image based on watershed and additional approaches in case of bad results
//this function is based on the method "watershed_one_box" defined below and it works in the following way and :
/*	
	1)	it tries to segment the current bounding box with a version of watershed based on the YCrCb color space,
		then it computes a binary image from the oringinal image using a threshold defined in the YCrCb color space,
		then it computes the intersection of these 2 results,
		then it computes the (number of white pixels of the final mask)/(number of total pixel of the mask)
		if the result of this operation is < 0.32 it passes to point 2), otherwise it keeps the current mask as final mask
	2)	it tries to segment the current bounding box with a version of watershed based on Otsu thresholding,
		then it computes a binary image from the oringinal image using a Otsu threshold,
		then it computes the intersection of these 2 results,
		then it computes the (number of white pixels of the final mask)/(number of total pixel of the mask)
		if the result of this operation is < 0.32 it passes to point 3), otherwise it keeps the current mask as final mask
	3)	it tries to segment the current bounding box with a version of watershed based on the HSV color space,
		then it computes a binary image from the oringinal image using a threshold in the HSV color space,
		then it computes the intersection of these 2 results,
		then it computes the (number of white pixels of the final mask)/(number of total pixel of the mask)
		if the result of this operation is < 0.32 it passes to point 4), otherwise it keeps the current mask as final mask
	4)	it tries to segment the current bounding box with K-means,
		then it computes the (number of white pixels of the final mask)/(number of total pixel of the mask)
		if the result of this operation is < 0.32 it passes to point 5), otherwise it keeps the current mask as final mask
	5)	if none of the masks obtained a value >= 0.32 it keeps the mask that obtained the biggest value
*/
//INPUT PARAMETERS: - the image (passed by const reference)
//                  - all of its bounding boxes (passed by const reference)
//OUPUT: a vector containing the masks of each hand in the image
std::vector<cv::Mat> my_watershed(const cv::Mat& img, const std::vector<Bounding_Box>& bboxes_img);


//function that applies the custom segmentation algorithm based on watershed and other support approaches
//INPUT PARAMETERS: - the image (passed by const reference)
//                  - the bounding box (passed by const reference)
//                  - the segmentation method to be used:
//                                          ---> if method == 0, then use wathershed and threshold on YCrCb color space combined
//                                          ---> if method == 1, then use wathershed and otsu combined
//                                          ---> if method == 2, then use wathershed and threshold on HSV color space combined
//OUTPUT: the mask obtained with this segmentation method
cv::Mat watershed_one_box(const cv::Mat& img, const Bounding_Box& box, const int& method = 0);

#endif