//author: Gabriele Boscarini, ID 2063145


#ifndef KMEANS_CLUSTERING_H
#define KMEANS_CLUSTERING_H

#include <opencv2/core.hpp>



void invert_pixels(cv::Mat& image);

int NoCorrelationCounter(const cv::Mat& image1, const cv::Mat& image2, int pixel1, int pixel2);

void treshold_otsu(cv::Mat& image);

//function that segment an image using k-means clustering
//INPUT PARAMETERS: -the image to be segmented
//                  -the number of clusters
//OUTPUT: a segmented image

cv::Mat K_Means(const cv::Mat& input, int k);

#endif
