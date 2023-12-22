Group members:
Edoardo Bastianello, ID 2053077
Stefano Binotto, ID 2052421
Gabriele Boscarini, ID 2063145

*********************************************************************************************
STRUCTURE OF THE DIRECTORIES:
- "src": it contains the code.
- "train_model": it contains the python script used to train the model (YOLOv5) , the .ipynb notebook version of the training, and 2 .yaml configuration file.
- the root directory contains: the report, the CMakeLists.txt and it also contain the the neural network model (.onnx format) if this is the version uploaded on the cloud.

Note: the trained model is the file "best.onnx"


*********************************************************************************************
The name of the executable obtained with CMake will be "project_hands".

THERE ARE 3 POSSIBILE WAYS TO TEST THE CODE (BASED ON THE DATASET YOU WANT TO USE):

1) TEST THE CODE WITH THE TEST DATASET OF 30 IMAGES THAT WAS ON GOOLE DRIVE:
To do this, you first need to place the "rgb" (with the test images in jpg format), "det" (with the txt files with the test bounding boxes) and "mask" (with the test masks in png format) folders in the same directory of the executable.
Next you have to specify from the command line the path to the trained model (.onnx file).

For example, you can execute the code with the following command (after you position in the directory with the executable):
./project_hands ../best.onnx



2) TEST THE CODE WITH A CUSTOM DATASET:
To do this, from the command line, you must specify the following parameters:
- the name of the executable
- the path to the trained .onnx model
- the directory with the test images (the name of the folder must be followed by "/")
- the directory with the test bounding boxes (the name of the folder must be followed by "/")
- the directory with the test masks (the name of the folder must be followed by "/")
- the format of the test images (with no dot ".")
- the format of the test masks (with no dot ".")

For example (after you position in the directory with the executable):
./project_hands ../best.onnx rgb/ det/ mask/ jpg png



3) TEST THE CODE WITH ONE SINGLE IMAGE
To do this, you must specify the following parameters from the command line:
- the name of the executable
- the path to the trained .onnx model
- the path to the test image
- the path to its corresponding test bounding boxes
- the path to its test mask

For example (after you position in the directory with the executable):
./project_hands ../best.onnx rgb/01.jpg det/01.txt mask/01.png