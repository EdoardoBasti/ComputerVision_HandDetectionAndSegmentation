//author: Edoardo Bastianello, ID:2053077

//file implementing the class Bounding_Box

#include <string>

#include <opencv2/core.hpp>

#include "Bounding_Box.h"


//***** CONSTRUCTOR *****
//INPUT PARAMETERS:	1-2) x, y coordinates of the top-left corner of the bounding box
//					3-4) width and height of the bounding box  
Bounding_Box::Bounding_Box(int x, int y, int width, int height)
{
	x_top_left = x;
	x_top_right = x + width - 1;	//example: if x = 0 and width is 3, it means that the x coordinate of the top right corner is 2 ((0, y), (1, y), (2,y) = three pixels -> width=3)
	x_bottom_left = x;
	x_bottom_right = x + width - 1;
	
	y_top_left = y;
	y_top_right = y;
	y_bottom_left = y + height - 1;
	y_bottom_right = y + height - 1;	
}


//function that returns the width of the bounding box
int Bounding_Box::get_width() const
{
	return x_top_right - x_top_left + 1;
}


//function that returns the height of the bounding box
int Bounding_Box::get_height() const
{
	return y_bottom_left - y_top_left + 1;
}


//function that, given a string as specified below, returns a point with the coordinates of the specified corner
//INPUT PARAMETERS: one of the following strings (passed by const reference):	-"top_left"
//																				-"top_right"
//																				-"bottom_left"
//																				-"bottom_right"
//OUTPUT:	-if one of these 4 strings is specified, then it returns a cv::Point2i with the 2 coordinates of the specified corner
//			-if another string is specified, then it returns a point with coordinates (-1, -1) 
cv::Point2i Bounding_Box::get_pair_xy(const std::string& position) const
{
	cv::Point2i point(-1, -1);

	if (!position.compare("top_left"))
	{
		point.x = x_top_left;
		point.y = y_top_left;
	}
	else if (!position.compare("top_right"))
	{
		point.x = x_top_right;
		point.y = y_top_right;
	}
	else if (!position.compare("bottom_left"))
	{
		point.x = x_bottom_left;
		point.y = y_bottom_left;
	}
	else if (!position.compare("bottom_right"))
	{
		point.x = x_bottom_right;
		point.y = y_bottom_right;
	}

	return point;
}