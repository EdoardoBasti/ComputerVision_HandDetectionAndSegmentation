//author: Edoardo Bastianello, ID:2053077

//functions to compute the values of Intersection over Union

#include <vector>

#include <opencv2/core.hpp>

#include "Bounding_Box.h"
#include "Intersection_over_Union.h"


//function that, given 2 integers, returns the max between them (integers passed by const reference)
int max(const int& x, const int& y)
{
	if (x > y)
	{
		return x;
	}
	return y;
}


//function that, given 2 integers, returns the min between them (integers passed by const reference)
int min(const int& x, const int& y)
{
	if (x < y)
	{
		return x;
	}
	return y;
}


//function that, given 2 object of the class Bounding_Box, returns the value of IoU (Intersection over Union)
//INPUT PARAMETERS: 2 objects of the class Bounding_Box (passed by const reference)
//OUTPUT: a double with the value of the intersection over union
double compute_intersection_over_union(const Bounding_Box& box1, const Bounding_Box& box2)
{
	int x_inter_1 = max(box1.get_pair_xy("top_left").x, box2.get_pair_xy("top_left").x);
	int y_inter_1 = max(box1.get_pair_xy("top_left").y, box2.get_pair_xy("top_left").y);

	int x_inter_2 = min(box1.get_pair_xy("bottom_right").x, box2.get_pair_xy("bottom_right").x);
	int y_inter_2 = min(box1.get_pair_xy("bottom_right").y, box2.get_pair_xy("bottom_right").y);

	//area intersection box
	int width_inter = max(0, x_inter_2 - x_inter_1 + 1);
	int height_inter = max(0, y_inter_2 - y_inter_1 + 1);
	double area_inter = width_inter * height_inter;

	//area union
	int width_box1 = box1.get_width();
	int height_box1 = box1.get_height();
	int width_box2 = box2.get_width();
	int height_box2 = box2.get_height();
	double area_box1 = width_box1 * height_box1;
	double area_box2 = width_box2 * height_box2;
	double area_union = area_box1 + area_box2 - area_inter;

	//intersection over union
	double inter_over_union = area_inter / area_union;

	//return inter_over_union;
	return inter_over_union;
}


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
std::vector<double> compute_IoU_of_one_img(const std::vector<Bounding_Box>& boxes_img_prediction, const std::vector<Bounding_Box>& boxes_img_test)
{
	//vector with values of IoU to be returned
	std::vector<double> intersections_over_unions_all_boxes;

	//vector of all distances between the predicted and test bounding boxes
	std::vector<std::vector<double>> distances;


	//compute distances
	for (int i = 0; i < boxes_img_prediction.size(); i++)
	{
		std::vector<double> distances_from_predicted_box;

		for (int j = 0; j < boxes_img_test.size(); j++)
		{
			double temp_dist = cv::norm(boxes_img_prediction[i].get_pair_xy("top_left") - boxes_img_test[j].get_pair_xy("top_left"));
			distances_from_predicted_box.push_back(temp_dist);
		}
		distances.push_back(distances_from_predicted_box);
	}
	
	//bounding boxes that are found both in the predicted vector and int the test vector
	int number_of_corrispondences = min(boxes_img_prediction.size(), boxes_img_test.size());


	//compute minimum distances
	for (int i = 0; i < number_of_corrispondences; i++)
	{
		std::vector<int> indices = min_distance_index(distances);
		int predicted = indices[0];
		int test = indices[1];

		distances[predicted][test] = DBL_MAX;

		double inter_over_union = compute_intersection_over_union(boxes_img_prediction[predicted], boxes_img_test[test]);
		intersections_over_unions_all_boxes.push_back(inter_over_union);
	}

	//bounding boxes which are found only in the predicted vector or only in the test vector
	int number_of_non_correspondences = max(boxes_img_prediction.size(), boxes_img_test.size()) - number_of_corrispondences;


	//for each extra/not found bounding box add a 0 to the IoU values
	for (int i = 0; i < number_of_non_correspondences; i++)
	{
		intersections_over_unions_all_boxes.push_back(0.0);
	}


	//return IoU values
	return intersections_over_unions_all_boxes;
}


//function that computes the minimum distance of a vector of vector and returns the indices of the minimum distance
//INPUT PARAMETERS: a vector of vector of double with the distances
//OUTPUT: the two indices with the minimum distance
std::vector<int> min_distance_index(const std::vector<std::vector<double>>& distances)
{
	double min_dist = DBL_MAX;
	int predicted = -1;
	int test = -1;

	//find minimum distance
	for (int i = 0; i < distances.size(); i++)
	{
		for (int j = 0; j < distances[i].size(); j++)
		{
			double temp_dist = distances[i][j];

			if (temp_dist < min_dist)
			{
				min_dist = temp_dist;
				predicted = i;
				test = j;
			}
		}
	}

	//return indices
	return { predicted, test };
}
