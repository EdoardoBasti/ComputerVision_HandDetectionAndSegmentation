//author: Stefano Binotto, ID:2052421

#ifndef PIXEL_ACCURACY_H
#define PIXEL_ACCURACY_H

#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>

/*
 * Compute Pixel Accuracy.
 * We assume that the masks are black and white with 3 color channels.
 * It takes in input the target binary mask and the detected
 * binary mask and gives in output the pixel accuracy.
 */
float pixel_accuracy(cv::Mat target_mask, cv::Mat detected_mask);

#endif