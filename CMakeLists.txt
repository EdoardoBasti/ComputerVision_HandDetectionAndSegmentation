cmake_minimum_required(VERSION 2.8)
project(project_hands)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/project_cv.cpp src/Bounding_Box.cpp src/Intersection_over_Union.cpp src/kmeans_clustering.cpp src/Mask_generator.cpp src/network_utils.cpp src/Pixel_Accuracy.cpp src/read_test_dataset.cpp src/segmentation.cpp src/visualize_results.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})