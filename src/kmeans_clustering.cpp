//author: Gabriele Boscarini, ID 2063145

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "kmeans_clustering.h"


using namespace std;
using namespace cv;

////function that, given an image it inverts the white color pixels with black color pixels and viceversa
//INPUT PARAMETERS: the image to be inverted 
//OUTPUT:  no output, it inverts the image


void invert_pixels(Mat& image) {

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			if (image.at<Vec3b>(i, j)[0] > 0 || image.at<Vec3b>(i, j)[1] > 0 || image.at<Vec3b>(i, j)[2] > 0) {

				image.at<Vec3b>(i, j)[0] = 0;
				image.at<Vec3b>(i, j)[1] = 0;
				image.at<Vec3b>(i, j)[2] = 0;
			}
			else {
				image.at<Vec3b>(i, j)[0] = 255;
				image.at<Vec3b>(i, j)[1] = 255;
				image.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
}

////function that given two images and two pixel intensities, calculate the number of pixels of image1 with intensity1 correspond to
//  pixels of image2 with intensity2 
// 
//INPUT PARAMETERS: two images and two intensietes
//OUTPUT:  the number of pixels of image1 with intensity1 that correspond at pixels of image2 with intensity2

int NoCorrelationCounter(const Mat& image1, const Mat& image2, int pixel1, int pixel2){

	int count = 0;

	//analize all pixels of the images

	for (int i = 0; i < image1.rows; i++) {
		for (int j = 0; j < image1.cols; j++) {

			//if pixel of image1 with intensity1 correspond to pixel of image2 with intensity2


			/*
			if ((image1.at<Vec3b>(i, j)[0] == pixel1 || image1.at<Vec3b>(i, j)[1] == pixel1 || image1.at<Vec3b>(i, j)[2] == pixel1)
				&& (image2.at<Vec3b>(i, j)[0] == pixel2 || image2.at<Vec3b>(i, j)[1] == pixel2 || image2.at<Vec3b>(i, j)[2] == pixel2)) {
				count++;
			}
			*/
			
			if ((image1.at<Vec3b>(i, j)[0] == pixel1 || image1.at<Vec3b>(i, j)[1] == pixel1 || image1.at<Vec3b>(i, j)[2] == pixel1)
				&& (image2.at<unsigned char>(i, j) == pixel2)) {
				count++;
			}
		}
	}
	return count;
}


//otsu's thresholding
//INPUT PARAMETERS: the image to be thresholded
//OUTPUT:  no output, it thresholds the image

void treshold_otsu(Mat& image) {
	
	GaussianBlur(image, image, Size(5, 5), 0, 0);
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	cv::threshold(image, image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
}



Mat K_Means(const Mat& in, int k) {

	cv::Mat input = in.clone();
	Mat comfront_image = in.clone();
	GaussianBlur(input, input, Size(15, 15), 0, 0);

	//comment this line if you want to use RGB space
	//cvtColor(input, input, COLOR_BGR2HSV);
	cvtColor(input, input, COLOR_BGR2YCrCb);

	Mat samples(input.rows * input.cols, input.channels(), CV_32F);
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			for (int z = 0; z < input.channels(); z++) {
				samples.at<float>(y + x * input.rows, z) = input.at<Vec3b>(y, x)[z];
			}
		}
	}

	Mat labels;
	int attempts = 15;
	Mat centers;

	kmeans(samples, k, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.001), attempts, KMEANS_PP_CENTERS, centers);

	Mat new_image(input.size(), input.type());
	for (int y = 0; y < input.rows; y++)
	{

		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input.rows, 0);
			if (input.channels() == 3)
			{
				for (int i = 0; i < input.channels(); i++)
				{
					if (cluster_idx == 0) {
						new_image.at<Vec3b>(y, x)[i] = 0;
					}
					else {
						new_image.at<Vec3b>(y, x)[i] = 255;
					}
				}
			}
		}
	}

	//convert a copy of the original image in ycrbr color space and then thresholds it
	treshold_otsu(comfront_image);


	//count1 is the number of black pixels in segmented image that are white pixels in thresholded image

	int count1 = NoCorrelationCounter(new_image, comfront_image, 0, 255);

	//count2 is the number of white pixels in segmented image that are white pixels in thresholded image

	int count2 = NoCorrelationCounter(new_image, comfront_image, 255, 255);

	if (count1 > count2) {

		//inverts the binary mask
		invert_pixels(new_image);

	}


	return new_image;
}
