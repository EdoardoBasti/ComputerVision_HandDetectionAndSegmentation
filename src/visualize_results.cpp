//author: Edoardo Bastianello, ID:2053077

//file that contains the functions to:
//visualize an image 
//visualize an image with its bounding boxes
//color the masks of an image, each one with a different color

#include <vector>
#include <string>
#include <stdexcept>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "Bounding_Box.h"
#include "visualize_results.h"


//function that visualizes an image
//INPUT PARAMETERS:	-the image to be visualized (passed by const reference)
//					-the name of the window (passed by const reference)
//OUTPUT: it visualizes an image, no value is returned
void visualize_image(const cv::Mat& img, const std::string& window_name)
{
	cv::namedWindow(window_name);

	cv::imshow(window_name, img);

	cv::waitKey(0);
}


//function that adds the bounding box passed as parameter to the image passed as parameter
//INPUT PARAMETERS:	-the image to which you want to add the bounding box (passed by reference)
//					-the bounding box to be added (passed by const reference)
//					-the color of the Bounding Box (one different color for each hand in the image)
//OUTPUT: None
void draw_one_bounding_box(cv::Mat& img_copy, const Bounding_Box& box, cv::Scalar color)
{
	int x_top_left = box.get_pair_xy("top_left").x;
	int y_top_left = box.get_pair_xy("top_left").y;

	int x_bottom_right = box.get_pair_xy("bottom_right").x;
	int y_bottom_right = box.get_pair_xy("bottom_right").y;

	//parameter one can change
	int line_thickness = 2;

	cv::rectangle(img_copy, cv::Point(x_top_left, y_top_left), cv::Point(x_bottom_right, y_bottom_right), color, line_thickness);
}


//function that visualizes an image with its bounding boxes
//INPUT PARAMETERS:	-the image to be visualized (passed by const reference)
//					-an std::vector of Bounding_Box objects (passed by const reference)
//OUTPUT: it visualizes an image, no value is returned
void visualize_results_bounding_box(const cv::Mat& img, const std::vector<Bounding_Box>& boxes)
{
	cv::Mat img_copy = img.clone();

	int number_of_bounding_boxes_of_the_image = boxes.size();

	//default color = RED
	cv::Scalar color = cv::Scalar(0, 0, 255);

	//seed for random generator
	//range 0, 255
	cv::RNG rng(12345);


	for (int i = 0; i < number_of_bounding_boxes_of_the_image; i++)
	{
		//random color
		//range: [0, 255]
		color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		draw_one_bounding_box(img_copy, boxes[i], color);
	}

	visualize_image(img_copy, "bounding_box_window");
}


//function that colors each mask in the vector with a different color
//INPUT PARAMETERS: a vector of masks (passed by reference)
//OUTPUT: None, it changes the colors
void color_masks(std::vector<cv::Mat>& img_masks)
{
	//default color = RED
	cv::Scalar color = cv::Scalar(0, 0, 255);

	//generator
	cv::RNG rng(12345);

	//for each hand mask of the image
	for (int i = 0; i < img_masks.size(); i++)
	{
		//randomly generate a color
		color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

		//current mask
		cv::Mat temp = img_masks[i];
		
		//for each pixel of the mask
		for (int row_index = 0; row_index < temp.rows; row_index++)
		{
			for (int col_index = 0; col_index < temp.cols; col_index++)
			{
				if (temp.channels() != 3)
				{
					throw std::invalid_argument("The image must have 3 channels\n");
				}
				//if 3 channels are not black
				if (temp.at<cv::Vec3b>(row_index, col_index)[0] > 0 || temp.at<cv::Vec3b>(row_index, col_index)[1] > 0 ||temp.at<cv::Vec3b>(row_index, col_index)[2] > 0)
				{
					//set the color to random color
					temp.at<cv::Vec3b>(row_index, col_index)[0] = color[0];
					temp.at<cv::Vec3b>(row_index, col_index)[1] = color[1];
					temp.at<cv::Vec3b>(row_index, col_index)[2] = color[2];
				}
			}
		}
	}
}