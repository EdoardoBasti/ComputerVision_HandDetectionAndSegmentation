//author: Edoardo Bastianello, ID:2053077

//file containing the class Bounding_Box

#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <string>

#include <opencv2/core.hpp>

class Bounding_Box
{
	public:
		//***** CONSTRUCTOR *****
		//INPUT PARAMETERS:	1-2) x, y coordinates of the top-left corner of the bounding box
		//					3-4) width and height of the bounding box 
		Bounding_Box(int x, int y, int width, int height);



		//function that, given a string as specified below, returns a point with the coordinates of the specified corner
		//INPUT PARAMETERS: one of the following strings (passed by const reference):	-"top_left"
		//																				-"top_right"
		//																				-"bottom_left"
		//																				-"bottom_right"
		//OUTPUT:	-if one of these 4 strings is specified, then it returns a cv::Point2i with the 2 coordinates of the specified corner
		//			-if another string is specified, then it returns a point with coordinates (-1, -1)
		cv::Point2i get_pair_xy(const std::string& position) const;



		//function that returns the width of the bounding box
		int get_width() const;



		//function that returns the height of the bounding box
		int get_height() const;


		
	private:
		//8 int (4 x and 4 y) which are the coordinates of the 4 corners of the bounding box
		int x_top_left, x_top_right, x_bottom_left, x_bottom_right;
		int y_top_left, y_top_right, y_bottom_left, y_bottom_right;
};

#endif