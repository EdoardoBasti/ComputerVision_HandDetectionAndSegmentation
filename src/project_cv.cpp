//authors:  Edoardo Bastianello, ID: 2053077
//          Stefano Binotto, ID: 2052421
//more details below on individual contributions

//this is the main for the project

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <opencv2/dnn.hpp>

//Edoardo Bastianello's files
#include "Bounding_Box.h"
#include "Intersection_over_Union.h"
#include "visualize_results.h"
#include "read_test_dataset.h"
#include "segmentation.h"

//Gabriele Boscarini's files
#include "kmeans_clustering.h"
#include "Mask_generator.h"

//Stefano Binotto's files
#include "network_utils.h"
#include "Pixel_Accuracy.h"


//if 5 command line arguments ----> executable - path to model -path to image - path to file with its bounding boxes - path to its mask
//if 7 command line arguments ----> executable - path to model - path to images directory - path to bboxes directory - path to masks directory - images format -masks format
//otherwise ----> read all 30 test images
int main(int argc, char** argv)
{
    //***************************************************
    //**** written by Edoardo Bastianello ***************
    //***************************************************

    std::cout << "************** OBJECT DETECTION MODULE **************\n";

    //images
    std::vector<cv::Mat> images;

    //executable - path to image - path to file with its bounding boxes - path to its mask
    if (argc == 5) //read only specified image
    {
        //read image from command line
        cv::Mat commandLineImg = cv::imread(argv[2]);
        
        //check if image is found
        if (commandLineImg.empty())
        {
            std::cout << "Could not open or find the image\n";
            return -1;
        }

        //put image in the vector of images
        images = { commandLineImg };
    }
    else //read multiple
    {
        try
        {
            if (argc == 7)
            {
                images = read_test_images(argv[2], argv[5]);
            }
            else
            {
                images = read_test_images("rgb/", "jpg");
            }
        }
        catch (std::invalid_argument e)//image not found
        {
            std::cout << e.what();
            return -1;
        } 
    }


    //***********************************************
    //**** written by Stefano Binotto ***************
    //***********************************************
    

    //path to model
    if (argc < 2)
    {
        std::cout << "you have to specify the path to the trained model\n";
        return -1;
    }
    std::string path_to_onnx = argv[1];


    //number of test images
    int s = images.size();


    //*********************************************************************
    //**** written by Edoardo Bastianello & Stefano Binotto ***************
    //*********************************************************************


    //predicted bounding boxes of all test images
    std::vector <std::vector<Bounding_Box>> predicted_boxes_all_images;

    std::cout << "** START PREDICTIONS **" << "\n";

    //sum of all intersection over union values
    double sum = 0.0;
    //sum of the intersection over union values without considering false negatives
    double sum_no_false_negatives = 0.0;

    //counter for all bounding boxes
    double counter = 0.0;
    //counter for bounding boxes without considering false negatives
    int counter_no_false_negatives = 0.0;


    //get ground truth boxes
    std::vector<std::vector<Bounding_Box>> test_boxes_all_images;
    if (argc == 5)
    {
        //get the bounding boxes of the specified image
        try
        {
            test_boxes_all_images = { read_Bounding_Boxes_one_img(argv[3]) };
        }
        catch (std::invalid_argument e)//file not found
        {
            std::cout << e.what();
            return -1;
        }
    }
    else
    {
        //get the bounding boxes of all images
        try
        {
            if (argc == 7)
            {
                test_boxes_all_images = read_test_b_box(argv[3]);
            }
            else
            {
                test_boxes_all_images = read_test_b_box("det/");
            }
        }
        catch (std::invalid_argument e)//file not found
        {
            std::cout << e.what();
            return -1;
        }
    }

    //check if number of files with bounding boxes is the same as the number of images
    if (test_boxes_all_images.size() != s)
    {
        std::cout << "the number of files with the bounding boxes doesn't correspond to the number of images\n";
        return -1;
    }




    for (int j = 0; j < s; j++) { //for each image

        //*************************************
        //**** written by Stefano Binotto *****
        //*************************************

        //current image
        cv::Mat img = images[j];

        //get predicted boxes
        std::vector<cv::Rect> bboxes = getBoundingBoxes(img, path_to_onnx);


        //************************************************************
        //**** written by Edoardo Bastianello and Stefano Binotto ****
        //************************************************************


        //bounding boxes of this image
        std::vector<Bounding_Box> boxes_img;


        //for each bounding box of this image
        for(int i=0; i < bboxes.size(); i++)
        {
            //construct Bounding_Box object
            Bounding_Box box = Bounding_Box(bboxes[i].x,bboxes[i].y,bboxes[i].width, bboxes[i].height);
            
            //add box to the bounding boxes of this image
            boxes_img.push_back(box);
        }
        

        //**********************************************
        //******* written by Edoardo Bastianello *******
        //**********************************************


        //get values of IoU for this image
        std::vector<double> inter_over_union_one_image = compute_IoU_of_one_img(boxes_img, test_boxes_all_images[j]);

        //print Intersection over Union values
        if (argc != 5)
        {
            std::cout << "IMAGE " << j + 1 << "\n";
        }
        else
        {
            std::cout << "IMAGE " << argv[2] << "\n";
        }

        //for each value of IoU
        for (int j = 0; j < inter_over_union_one_image.size(); j++)
        {
            //update sum
            sum = sum + inter_over_union_one_image[j];

            //update counter
            counter++;

            //case where we don't consider false negatives
            if (inter_over_union_one_image[j] != 0)
            {
                //update sum
                sum_no_false_negatives = sum_no_false_negatives + inter_over_union_one_image[j];

                //update counter
                counter_no_false_negatives++;
            }

            //print value
            std::cout << "IoU = " << inter_over_union_one_image[j] << "\n";
        }

        std::cout << "\n";

        //visualize bounding boxes
        visualize_results_bounding_box(img, boxes_img);


        //add the bounding boxes of this image to all the predicted bounding boxes
        predicted_boxes_all_images.push_back(boxes_img);
    }
    std::cout << "** END PREDICTIONS **" << "\n";


    //**********************************************
    //******* written by Edoardo Bastianello *******
    //**********************************************


    //compute average Intersection over Union
    double average_inter_over_union = sum / counter;
    double average_inter_over_union_no_false_negatives = sum_no_false_negatives / counter_no_false_negatives;

    //print average Intersection over Union result
    std::cout << "AVERAGE IoU: " << average_inter_over_union << "\n";
    std::cout << "AVERAGE IoU WITHOUT FALSE NEGATIVES: " << average_inter_over_union_no_false_negatives << "\n";







    cv::destroyAllWindows();
    std::cout << "\n************** SEGMENTATION MODULE **************\n";


    //test masks
    std::vector<cv::Mat> test_masks_all_images;
    if (argc == 5)//then read specified mask
    {
        //read mask from command line
        cv::Mat commandLineMask = cv::imread(argv[4]);

        //check if mask is found
        if (commandLineMask.empty())
        {
            std::cout << "Could not open or find the mask\n";
            return -1;
        }

        //put image in the vector of masks
        test_masks_all_images = { commandLineMask };
    }
    else//then read all 30 masks
    {
        try
        {
            if (argc == 7)
            {
                test_masks_all_images = read_test_images(argv[4], argv[6]);
            }
            else
            {
                test_masks_all_images = read_test_images("mask/", "png");
            }
        }
        catch (std::invalid_argument e)//mask not found
        {
            std::cout << e.what();
            return -1;
        }
    }

    //check if number of masks is equal to the number of images
    int s_masks = test_masks_all_images.size();
    if (s != s_masks)
    {
        std::cout << "the number of masks doesn't correspond to the number of images\n";
        return -1;
    }

    

    
    //************************************************************
    //**** written by Edoardo Bastianello and Stefano Binotto ****
    //************************************************************

    //initialization average accuracy
    float average_accuracy = 0.0;
    for (int index = 0; index < s; index++) //for each image
    {
        if (argc != 5)
        {
            std::cout << "IMAGE " << index + 1 << ":\n";
        }
        else
        {
            std::cout << "IMAGE " << argv[2] << "\n";
        }

        //current image
        cv::Mat img_to_segment = images[index];

        //current predicted boxes of the current image
        std::vector<Bounding_Box> boxes_img_to_seg = predicted_boxes_all_images[index];

        //apply custom watershed
        std::vector<cv::Mat> water_masks_img = my_watershed(img_to_segment, boxes_img_to_seg);

       
        //get colored masks
        cv::Mat colored_mask;
        try
        {
            colored_mask = getFinalMask(img_to_segment, water_masks_img, boxes_img_to_seg, "colored");
        }
        catch (std::invalid_argument e)
        {
            std::cout << e.what();
        }
       
        //get binary masks
        cv::Mat binary_mask = getFinalMask(img_to_segment, water_masks_img, boxes_img_to_seg, "binary");

	    //get accuracy
        float accuracy = 0.0;
	    try
        {
        	 accuracy = pixel_accuracy(test_masks_all_images[index], binary_mask);
        }
        catch (std::invalid_argument e)	//non-valid input arguments 
        {
            std::cout << e.what();
            return -1;
        }
        
        //update sum for average accuracy
        average_accuracy = average_accuracy + accuracy;

        //print accuracy of this mask
        std::cout << "PIXEL ACCURACY: " << accuracy << "\n";

        //visualize resut
        visualize_image(colored_mask, "segmentation");
    }

    //compute average accuracy
    average_accuracy = average_accuracy / s;

    //print value of average accuracy
    std::cout << "\nAVERAGE ACCURACY: " << average_accuracy << "\n";

    return 0;
}
