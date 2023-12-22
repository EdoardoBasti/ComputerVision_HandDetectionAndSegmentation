//author Gabriele Boscarini, ID 2063145

#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "visualize_results.h"
#include "Bounding_Box.h"
#include "Mask_generator.h"
#include "segmentation.h"

using namespace std;
using namespace cv;


//function that overlap masks in regions of interest of an image
//INPUT PARAMETERS:	-the image before paste masks (passed by const reference)
//					-the image in which masks are pasted 
//                  -the bounding boxes in wich operate the overlap
//OUTPUT: it overlaps an image, no value is returned

void overlap_image(const Mat& input, Mat& coloredImage, vector<Bounding_Box> boxes, const vector<Mat>& masks) {


	//for each bounding box

	for (int i = 0; i < boxes.size(); i++) {

		Point2d pair_xy = boxes[i].get_pair_xy("top_left");

		//for each pixel of the roi of the mask

		for (int y = 0; y < masks[i].rows; y++) {
			for (int x = 0; x < masks[i].cols; x++) {

				//if the pixel is black

				if (masks[i].at<Vec3b>(y, x)[0] != 0 || masks[i].at<Vec3b>(y, x)[1] != 0 ||
					masks[i].at<Vec3b>(y, x)[2] != 0) {

					//substitute the pixel with the pixel of the image without mask

					coloredImage.at<Vec3b>(pair_xy.y + y, pair_xy.x +x)[0] = masks[i].at<Vec3b>(y, x)[0];
					coloredImage.at<Vec3b>(pair_xy.y + y, pair_xy.x + x)[1] = masks[i].at<Vec3b>(y, x)[1];
					coloredImage.at<Vec3b>(pair_xy.y + y, pair_xy.x + x)[2] = masks[i].at<Vec3b>(y, x)[2];
				}
			}
		}
	}
}

//function that return two possible final masks: binary or colored
//INPUT PARAMETERS:	-the initial image
//                  -vector containing masks images as result of segmentation
//                  -the bounding boxes in wich segmentation was made
//                  - a string for choice wich mask is the output: "binary" for binary mask or "colored" for colored mask
//OUTPUT: return the final mask


Mat getFinalMask(const Mat& in, vector<Mat>& masks, vector<Bounding_Box> boxes, string MaskType) {

	//make a copy of original image to create final mask

	Mat image = in.clone();
	Mat colored_mask = in.clone();

	vector<Mat> masks_copy;
	for (int i = 0; i < masks.size(); i++)
	{
		masks_copy.push_back(masks[i].clone());
	}

	// create black mask

	Mat binary_mask(in.rows, in.cols, CV_8UC3, Scalar(0, 0, 0));
	Mat final_image;

	//create binary mask

	if (!MaskType.compare("binary")) {

		overlap_image(binary_mask, binary_mask, boxes, masks_copy);
		final_image = binary_mask;
	}

	//create colored mask

	else if (!MaskType.compare("colored")) {

		color_masks(masks_copy);
		overlap_image(image, colored_mask, boxes, masks_copy);
		final_image = colored_mask;
	}

	return final_image;
}



