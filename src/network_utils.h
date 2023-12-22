//author: Stefano Binotto, ID:2052421

#ifndef NETWORK_UTILS_H
#define NETWORK_UTILS_H

#include<vector>
#include<string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

/*
 * Preprocessing used to convert the image into a 
 * blob and set it up for the network. It takes in input 
 * the image in which we want to detect the boxes, and outputs 
 * the blob that can be accepted by the model. 
 */
cv::Mat preprocessing(const cv::Mat &source);

/*
 * Prediction of the model. It takes in input the blob 
 * gotten in the "preprocessing()" method and gives in output the cv::Mat
 * containing all the bounding boxes detected.
 */
cv::Mat detect(const cv::Mat &input_blob, cv::dnn::Net &net);

/*
 * Postprocessing is used to unwrapped the detection result cv::Mat for further 
 * processing:
 * -Find all the detections
 * -Keep only the best detections, discarding detections with confidence lower than a threshold value
 * -Non-Maxima-Suppression to delete the multiple overlapping boxes
 * It takes in input the original image and the detections gotten by the "detect()" method
 * and gives in output the final boxes detected.
 */
std::vector<cv::Rect> postprocessing(const cv::Mat &input_image, cv::Mat &detections);

/*
 *	Method used to get a vector containing the bounding boxes inside the "input_image".
 *  It requires the path to the .onnx file as input argument and the reference to the 
 * 	original image.
 */
std::vector<cv::Rect> getBoundingBoxes(const cv::Mat &input_image, std::string path_to_onnx);

#endif