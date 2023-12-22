//author: Stefano Binotto, ID:2052421

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "Pixel_Accuracy.h"

/*
 * Compute Pixel Accuracy.
 * We assume that the masks are black and white with 3 color channels.
 * It takes in input the target binary mask and the detected
 * binary mask and gives in output the pixel accuracy.
 */
float pixel_accuracy(cv::Mat target_mask, cv::Mat detected_mask) {
	
	//store size of the two masks
	int rows_target = target_mask.rows;
	int rows_detected = detected_mask.rows;
	int cols_target = target_mask.cols;
	int cols_detected = detected_mask.cols;
	
	//check whether the two masks have the same size
	if ((rows_target != rows_detected) || (cols_target != cols_detected)) {
        throw std::invalid_argument( "The target segmentation mask and the detected segmentation mask must have the same size!!!" );
    }
	
	//the function works if the masks are 3 channels cv::Mat
	if ((target_mask.channels() != 3) || (detected_mask.channels() != 3)) {
        throw std::invalid_argument( "The segmentation masks must have 3 channels!!!" );
    }
	
	float TP = 0;	//True Positive
	float TN = 0;	//True Negative
	float FP = 0;	//False Positive
	float FN = 0;	//False Negative
	
	//compare each pixel for both the masks
	for(int i = 0; i < rows_detected; i++) {
		for(int j = 0; j < cols_detected; j++) {
			
			//store the BGR values of the detected pixel
			int det_B = detected_mask.at<cv::Vec3b>(i,j)[0];
			int det_G = detected_mask.at<cv::Vec3b>(i,j)[1];
			int det_R = detected_mask.at<cv::Vec3b>(i,j)[2];
			
			//store the BGR values of the target pixel
			int tar_B = target_mask.at<cv::Vec3b>(i,j)[0];
			int tar_G = target_mask.at<cv::Vec3b>(i,j)[1];
			int tar_R = target_mask.at<cv::Vec3b>(i,j)[2];
			
			//check whether the detected mask is black and white
			if ((det_B != 255 || det_G != 255 || det_R != 255) && (det_B != 0 || det_G != 0 || det_R != 0)) {
				throw std::invalid_argument( "The detected segmentation mask must be black and white!!!" );
			}
			
			//check whether the target mask is black and white
			if ((tar_B != 255 || tar_G != 255 || tar_R != 255) && (tar_B != 0 || tar_G != 0 || tar_R != 0)) {
				throw std::invalid_argument( "The target segmentation mask must be black and white!!!" );
			}
			
			if (det_B == 255 && det_G == 255 && det_R == 255) {	  //if the pixel in the detected mask is white
				if (tar_B == 255 && tar_G == 255 && tar_R == 255) {		//if the pixel in the target mask is white
					TP++;
				} else if (tar_B == 0 && tar_G == 0 && tar_R == 0) {	//if the pixel in the target mask is black
					FP++;
				}
			} else if (det_B == 0 && det_G == 0 && det_R == 0) {	//if the pixel in the detected mask is black
				if (tar_B == 255 && tar_G == 255 && tar_R == 255) {		//if the pixel in the target mask is white
					FN++;
				} else if (tar_B == 0 && tar_G == 0 && tar_R == 0) {	//if the pixel in the target mask is black
					TN++;
				}
			}
		}
	}
	
	//(number of hand pixels classified as hand) + (number of non-hand pixels classified as non-hand)
	float Num = TP+TN;
	
	//total number of pixels
	float Den = TP+TN+FP+FN;
	
	//compute Pixel Accuracy of the segmentation
	float accuracy = Num/Den;
	
	return accuracy;
}