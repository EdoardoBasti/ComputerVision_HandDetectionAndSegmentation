//author: Edoardo Bastianello, ID:2053077

//functions to compute the values of Intersection over Union

#ifndef INTERSECTION_OVER_UNION_H
#define INTERSECTION_OVER_UNION_H

#include <vector>

#include "Bounding_Box.h"


//function that, given 2 integers, returns the max between them (integers passed by const reference)
int max(const int& x, const int& y);


//function that, given 2 integers, returns the min between them (integers passed by const reference)
int min(const int& x, const int& y);


//function that, given 2 object of the class Bounding_Box, returns the value of IoU (Intersection over Union)
//INPUT PARAMETERS: 2 objects of the class Bounding_Box (passed by const reference)
//OUTPUT: a double with the value of the intersection over union
double compute_intersection_over_union(const Bounding_Box& box1, const Bounding_Box& box2);


//function that, given the bounding boxes of 2 images, returns the values of IoU
//this functions solves the following 2 problems:
//	1) What if, for a given test image, our network finds a number of bounding boxes lower or greater than the number of bounding boxes of the test dataset?
//			(if number of predicted bounding boxes < number of true bounding boxes)
//					----> returns one value of IoU for each predicted bounding box and 0.0 for each non predicted bounding box that is in the ground truth
//			(if number of predicted bounding boxes > number of true bounding boxes)
//					----> returns one value of IoU for each predicted bounding box and 0.0 for the extra bounding boxes
//	2) Which of the test bounding boxes a predicted bounding box corresponds to?
//		---> By defining the origin of a Bounding Box as its top left vertex, a predicted Bounding Box corresponds to the test Bounding Box with minimum distance between their origins
//INPUT PARAMETERS: 2 vectors of bounding box; each one contains all the bounding boxes of an image
//OUTPUT: a vector of Intersection over Union values
std::vector<double> compute_IoU_of_one_img(const std::vector<Bounding_Box>& boxes_img_prediction, const std::vector<Bounding_Box>& boxes_img_test);


//function that computes the minimum distance of a vector of vector and returns the indices of the minimum distance
//INPUT PARAMETERS: a vector of vector of double with the distances
//OUTPUT: the two indices with the minimum distance
std::vector<int> min_distance_index(const std::vector<std::vector<double>>& distances);


#endif