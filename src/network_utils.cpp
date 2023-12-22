//author: Stefano Binotto, ID:2052421

#include<vector>
#include<string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "network_utils.h"

//width and height of the input blob for the model
const float Width = 640.0;
const float Height = 640.0; 

/*
 * The following two thresholds are found by testing all the 
 * images in the testset. Their values are the ones that can approximate a good average
 * detection over all the images.
 */

//confidence parameter to keep only the best detections
const float CONFIDENCE_THRESHOLD = 0.21;

//Non-Maxima Suppression parameter to delete the multiple overlapping boxes
const float NMS_THRESHOLD = 0.425;
 
/*
 * Preprocessing used to convert the image into a 
 * blob and set it up for the network. It takes in input 
 * the image in which we want to detect the boxes, and outputs 
 * the blob that can be accepted by the model. 
 */
cv::Mat preprocessing(const cv::Mat &source) {

	/*
	 * Yolov5 accepts input blob of size 640x640, 
	 * with pixel values in [0,1], and RGB format
	 * (openCV use BGR, so I need to swap the channels)
	 */

	//convert the resized image into a blob
	cv::Mat blob;
	
	//create the blob with [0, 1] normalization, resized dimensions and switch from BGR to RGB
	cv::dnn::blobFromImage(source, blob, 1./255., cv::Size(Width, Height), cv::Scalar(), true, false); //true = swap R & B channels
	
	return blob;
}

/*
 * Prediction of the model. It takes in input the blob 
 * gotten in the "preprocessing()" method and gives in output the cv::Mat
 * containing all the bounding boxes detected.
 */
cv::Mat detect(const cv::Mat &input_blob, cv::dnn::Net &net) {
	
	//store all the detections of the network on input_blob
	std::vector<cv::Mat> detections;
	
	//set the new input value for the network
	net.setInput(input_blob);
	
	//forward pass
	net.forward(detections, net.getUnconnectedOutLayersNames());
	
	return detections[0];
}

/*
 * Postprocessing is used to unwrapped the detection result cv::Mat for further 
 * processing:
 * -Find all the detections
 * -Keep only the best detections, discarding detections with confidence lower than a threshold value
 * -Non-Maxima-Suppression to delete the multiple overlapping boxes
 * It takes in input the original image and the detections gotten by the "detect()" method
 * and gives in output the final boxes detected.
 */
std::vector<cv::Rect> postprocessing(const cv::Mat &input_image, cv::Mat &detections) {
	
	//compute resize factor
	float factor_x = static_cast<float>(input_image.cols)/Width;
	float factor_y = static_cast<float>(input_image.rows)/Height;
	
	//store the detected bounding boxes and the respective confidence values
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	
	/*
	 * the detections matrix has 25200 rows (one per detection),
	 * each row has [x_center, y_center, width, height, confidence, score for class hand]
	 */ 
	
	//number of detections for 640x640 images ---> 25200
	const int num_detections = 25200;  

	//Iterate through all the 25200 detections
	for(int i = 0; i < num_detections; i++) {
		//study the detected box i
		
		//store the confidence of the box i
		float confidence = detections.at<float>(0, i, 4);
		
		//check whether the box i has a good confidence value
		if(confidence >= CONFIDENCE_THRESHOLD) {
			
			//store confidence of the detection for Non-Maxima-Suppression
			confidences.push_back(confidence);
			
			//box center coordinates and size with respect to the 640x640 blob
			float center_x = detections.at<float>(0, i, 0);
			float center_y = detections.at<float>(0, i, 1);
			float norm_width = detections.at<float>(0, i, 2);
			float norm_height = detections.at<float>(0, i, 3);
			
			//box coordinates with respect to the original dimensions
			int x = static_cast<int>((center_x-(norm_width/2))*factor_x);
			int y = static_cast<int>((center_y-(norm_height/2))*factor_y);
			int w = static_cast<int>(norm_width*factor_x);
			int h = static_cast<int>(norm_height*factor_y);
			
			//store bbox
			boxes.push_back(cv::Rect(x, y, w, h));
		}
	}
	
	//Non-Maxima-Suppression to remove multiple overlapping boxes
	std::vector<int> indices;
	std::vector<cv::Rect> Bboxes;
	
	//I use the CONFIDENCE_THRESHOLD and the NMS_THRESHOLD that I defined above 
	//in order to refine the detection
	cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
	
	for (int i = 0; i < indices.size(); i++) {
		//push in the vector of the final detected boxes only the ones which correspond to the indeces in the vector "indices" 
		int index = indices[i];
		Bboxes.push_back(boxes[index]);
	}
	
	return Bboxes;	
}

/*
 *	Method used to get a vector containing the bounding boxes inside the "input_image".
 *  It requires the path to the .onnx file as input argument and the reference to the 
 * 	original image.
 */
std::vector<cv::Rect> getBoundingBoxes(const cv::Mat &input_image, std::string path_to_onnx) {
	//prepocessing of the input image
	cv::Mat preprocessed = preprocessing(input_image);
	
	//import the model
	cv::dnn::Net net = cv::dnn::readNet(path_to_onnx);
	
	//DECOMMENT THE FOLLOWING 2 LINES IF NVIDIA CARD WITH CUDA IS AVAILABLE
	//(if CUDA is not available, the functions automatically switch CPU execution)
	/*
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	*/
	
	//detect all possible boxes
	cv::Mat detections = detect(preprocessed, net);
	
	//refine the detections
	std::vector<cv::Rect> bboxes = postprocessing(input_image, detections);
	
	return bboxes;
}