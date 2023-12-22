//author: Edoardo Bastianello, ID:2053077

//file to segment an image on a custom approach based on watershed intersected with a binary image (and other methods if the results obtained with watershed are not good enough)

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "Bounding_Box.h"
#include "segmentation.h"
#include "kmeans_clustering.h"


//function that counts how many white pixels there are in an image
//INPUT PARAMETERS: the image (passed by const reference)
//OUTPUT: an integer with the number of white pixels in the image
int count_white_pixels_img(const cv::Mat& img)
{
    //initialize sum
    int sum_white = 0;

    //update count
    for (int j = 0; j < img.rows; j++)
    {
        for (int z = 0; z < img.cols; z++)
        {
            if (img.at<unsigned char>(j, z) > 0)
            {
                sum_white++;
            }
        }
    }

    //return count
    return sum_white;
}


//function that, given an image and one of its bounding boxes, returns a cropped image with the region contained in the bounding box
//INPUT PARAMETERS: - the image (passed by const reference)
//                  - one of its bounding boxes (passed by const reference)
//OUTPUT: a cropped image containing the region of the image inside the bounding box
cv::Mat crop_Bounding_Box(const cv::Mat& img, const Bounding_Box& bbox)
{
    //get cropped region
	cv::Rect cropped_region(bbox.get_pair_xy("top_left").x, bbox.get_pair_xy("top_left").y, bbox.get_width(), bbox.get_height());
	cv::Mat cropped_img = img(cropped_region);

    //return cropped region
	return cropped_img;
}


//function that computes the segmentation of one image based on watershed and additional approaches in case of bad results
//this function is based on the method "watershed_one_box" defined below and it works in the following way and :
/*
    1)	it tries to segment the current bounding box with a version of watershed based on the YCrCb color space,
        then it computes a binary image from the oringinal image using a threshold defined in the YCrCb color space,
        then it computes the intersection of these 2 results,
        then it computes the (number of white pixels of the final mask)/(number of total pixel of the mask)
        if the result of this operation is < 0.32 it passes to point 2), otherwise it keeps the current mask as final mask
    2)	it tries to segment the current bounding box with a version of watershed based on Otsu thresholding,
        then it computes a binary image from the oringinal image using a Otsu threshold,
        then it computes the intersection of these 2 results,
        then it computes the (number of white pixels of the final mask)/(number of total pixel of the mask)
        if the result of this operation is < 0.32 it passes to point 3), otherwise it keeps the current mask as final mask
    3)	it tries to segment the current bounding box with a version of watershed based on the HSV color space,
        then it computes a binary image from the oringinal image using a threshold in the HSV color space,
        then it computes the intersection of these 2 results,
        then it computes the (number of white pixels of the final mask)/(number of total pixel of the mask)
        if the result of this operation is < 0.32 it passes to point 4), otherwise it keeps the current mask as final mask
    4)	it tries to segment the current bounding box with K-means,
        then it computes the (number of white pixels of the final mask)/(number of total pixel of the mask)
        if the result of this operation is < 0.32 it passes to point 5), otherwise it keeps the current mask as final mask
    5)	if none of the masks obtained a value >= 0.32 it keeps the mask that obtained the biggest value
*/
//INPUT PARAMETERS: - the image (passed by const reference)
//                  - all of its bounding boxes (passed by const reference)
//OUPUT: a vector containing the masks of each hand in the image
std::vector<cv::Mat> my_watershed(const cv::Mat& img, const std::vector<Bounding_Box>& bboxes_img)
{
    //vector of masks to be returned
    std::vector<cv::Mat> final_masks_img;

    //for each bounding box of the image
    for (int i = 0; i < bboxes_img.size(); i++)
    {
        //mask result of this box
        cv::Mat result;

        //iitialize parameters to choose the best approach for segmentation
        int iterations = 0; //0, 1, 2, 3 ----> each one tries a different segmentation approach
        double threshold = 0.32; //if white_pixels_of_the_mask/all_pixels_of_the_mask >= threshold ---> this method is probably good
        double area = 0.0; //all_pixels_of_the_mask
        double sum_white = 0.0;  //white_pixels_of_the_mask
        double white_over_area = 0.0; //white_pixels_of_the_mask/all_pixels_of_the_mask

        //area of best approach (the larger the number of white_pixels_of_the_mask the better)
        double best_area_results = -1.0;

        //index of best approach
        //if 0 -> watershed + YCrCb color space
        //if 1 -> watershed + otsu
        //if 2 -> watershed + HSV color space
        //if 3 -> kmeans
        int best_method = -1;

        //masks of this box, one for each approach
        std::vector<cv::Mat> results_one_bbox;

        while (white_over_area < threshold && iterations < 4)
        {
            if (iterations < 3)//use one of the custom variants of watershed
            {
                //the integer in "iterations" is the index of the segmentation method to be used
                result = watershed_one_box(img, bboxes_img[i], iterations);
            }
            else //if watershed fails then use kmeans
            {
                //get cropped region
                cv::Mat taglia = crop_Bounding_Box(img, bboxes_img[i]);

                //apply kmeans
                result = K_Means(taglia, 2);
            }
            

            //compute white pixels over all pixels of the mask
            area = result.rows * result.cols;
            sum_white = count_white_pixels_img(result);
            white_over_area = sum_white / area;

            //add the mask to the possible masks of this approach
            results_one_bbox.push_back(result);

            //update current best method
            if (white_over_area > best_area_results)
            {
                best_area_results = white_over_area;
                best_method = iterations;
            }

            //update number o interaction and change approach
            iterations++;
        }

        //best mask
        cv::Mat final_best = results_one_bbox[best_method];

        //convert to right format (type 16)
        if (final_best.channels() == 1)
        {
            cv::cvtColor(final_best, final_best, cv::COLOR_GRAY2BGR);
            final_best.convertTo(final_best, CV_8UC3);
        }
        //add best mask to the vector of masks
        final_masks_img.push_back(final_best);
    }

    //return masks (one for each hand)
    return final_masks_img;
}


//function that applies the custom segmentation algorithm based on watershed and other support approaches
//INPUT PARAMETERS: - the image (passed by const reference)
//                  - the bounding box (passed by const reference)
//                  - the segmentation method to be used:
//                                          ---> if method == 0, then use wathershed and threshold on YCrCb color space combined
//                                          ---> if method == 1, then use wathershed and otsu combined
//                                          ---> if method == 2, then use wathershed and threshold on HSV color space combined
//OUTPUT: the mask obtained with this segmentation method
cv::Mat watershed_one_box(const cv::Mat& img, const Bounding_Box& box, const int& method)
{
    //try to icrease the size of the box by an amount of pixels equals to "bigger" in each direction
    int bigger = 4;
    int larger_left = 0;
    int larger_right = 0;

    int higher_top = 0;
    int lower_bottom = 0;

    //find maximum increment possibile (if a bounding box is on the edge of the image then don't icrement it too much)
    for (int j = 0; j <= bigger; j++)
    {
        if ((box.get_pair_xy("top_left").x - j) >= 0)
        {
            larger_left = j;
        }
        if ((box.get_pair_xy("top_left").y - j) >= 0)
        {
            higher_top = j;
        }
        if ((box.get_pair_xy("top_right").x + j) <= (img.cols - 1))
        {
            larger_right = j;
        }
        if ((box.get_pair_xy("bottom_left").y + j) <= (img.rows - 1))
        {
            lower_bottom = j;
        }
    }

    //get larger box
    Bounding_Box larger_box = Bounding_Box(box.get_pair_xy("top_left").x - larger_left, box.get_pair_xy("top_left").y - higher_top, box.get_width() + larger_left + larger_right, box.get_height() + higher_top + lower_bottom);
    
    //crop image keeping the region inside the larger box
    cv::Mat cropped = crop_Bounding_Box(img, larger_box);
    
    //copy source img
    cv::Mat src = cropped.clone();

    //create a kernel (similar to Laplacian)
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);

    //sharpen the image usig the kernel
    cv::Mat imgLaplacian;
    cv::filter2D(src, imgLaplacian, CV_32F, kernel);
    cv::Mat sharp;
    src.convertTo(sharp, CV_32F);
    cv::Mat imgResult = sharp - imgLaplacian;
    //convert to 8 bits grayscale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);


    //obtain binary image using the selected method
    cv::Mat bw;
    if (!method)
    {
        cv::cvtColor(imgResult, bw, cv::COLOR_BGR2YCrCb);
        cv::inRange(bw, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), bw);
    }
    else if (method == 1)
    {
        cv::cvtColor(imgResult, bw, cv::COLOR_BGR2GRAY);
        cv::threshold(bw, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }
    else
    {
        cv::cvtColor(imgResult, bw, cv::COLOR_BGR2HSV);
        cv::inRange(bw, cv::Scalar(0, 0.28 * 255, 0), cv::Scalar(25, 0.68 * 255, 255), bw);
    }

    //commpute distance tranform
    cv::Mat dist;
    cv::distanceTransform(bw, dist, cv::DIST_L2, 3);

    //normalize distances in [0,1] to be able to threshold them
    cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);


    //threshold distances to obtain peaks that will be the foreground markers (hand markers)
    double dist_threshold = 0.5;
    cv::threshold(dist, dist, dist_threshold, 1.0, cv::THRESH_BINARY);


    //dilate the obtained peak image
    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
    cv::dilate(dist, dist, kernel1);



    //CV_8U version of the distance image to be able to use findContours()
    cv::Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    //find markers
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


    //find largest contour ---> it will probably be the palm of the hand
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int z = 0; z < contours.size(); z++) {
        double newArea = cv::contourArea(contours.at(z));
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = z;
        }
    }


    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);

    //draw foreground markers
    cv::drawContours(markers, contours, maxAreaContourId, cv::Scalar(maxAreaContourId + 1), -1, 0);


    //background markers ---> just outside the bounding box, one for each corner
    int radius = 1;
    cv::circle(markers, cv::Point(0, 0), 1, cv::Scalar(255), -1);
    cv::circle(markers, cv::Point(cropped.cols - 1, 0), radius, cv::Scalar(255), -1);
    cv::circle(markers, cv::Point(0, cropped.rows - 1), radius, cv::Scalar(255), -1);
    cv::circle(markers, cv::Point(cropped.cols - 1, cropped.rows - 1), radius, cv::Scalar(255), -1);
    //top & bottom ---> one for each 30 pixels
    for (int i = 0; i < cropped.cols; i++)
    {
        if (i % 30 == 0)
        {
            cv::circle(markers, cv::Point(i, 0), radius, cv::Scalar(255), -1);
            cv::circle(markers, cv::Point(i, cropped.rows - 1), radius, cv::Scalar(255), -1);
        }
    }
    //left & right ---> one for each 30 pixels
    for (int i = 0; i < cropped.rows; i++)
    {
        if (i % 30 == 0)
        {
            cv::circle(markers, cv::Point(0, i), radius, cv::Scalar(255), -1);
            cv::circle(markers, cv::Point(cropped.cols - 1, i), radius, cv::Scalar(255), -1);
        }
    }

    cv::Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);


    //watershed
    cv::watershed(imgResult, markers);

    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);


    //if a pixel > 0 ---> set it to 255
    for (int row = 0; row < mark.rows; row++)
    {
        for (int col = 0; col < mark.cols; col++)
        {
            if (mark.at<unsigned char>(row, col) > 0)
            {
                mark.at<unsigned char>(row, col) = 255;
            }
        }
    }


    //intersect the image obtained with watershed with the binary image obtained in the method specified by function parameter "method"
    cv::Mat inter;
    if (!method)  //intersect with YCrCb threshold
    {
        cv::Mat ycrcb;
        cv::cvtColor(cropped, ycrcb, cv::COLOR_BGR2YCrCb);
        cv::inRange(ycrcb, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), ycrcb);
        cv::bitwise_and(mark, ycrcb, inter);
    }
    else if (method == 1)   //intersect with otsu threshold
    {
        cv::Mat otsu;
        cv::cvtColor(cropped, otsu, cv::COLOR_BGR2GRAY);
        cv::threshold(otsu, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::bitwise_and(mark, otsu, inter);
    }
    else //intersect with HSV threshold
    {
        cv::Mat hsv;
        cv::cvtColor(cropped, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 0.28 * 255, 0), cv::Scalar(25, 0.68 * 255, 255), hsv);
        cv::bitwise_and(mark, hsv, inter);
    }


    //crop the larger bounding box to match the size of the right bounding box
    Bounding_Box original_size(larger_left, higher_top, box.get_width(), box.get_height());
    cv::Mat img_original_size = crop_Bounding_Box(inter, original_size);

    //return mask
    return img_original_size;
}