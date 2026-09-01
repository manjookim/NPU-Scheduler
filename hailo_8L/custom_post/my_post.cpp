#include <iostream>
#include "my_post.hpp"

std::vector<HailoDetection> demo_detection_objects()
{
    std::vector<HailoDetection> objects; // The detection objects we will eventually return
    HailoDetection first_detection = HailoDetection(HailoBBox(0.2, 0.2, 0.2, 0.2), "person", 0.99);
    HailoDetection second_detection = HailoDetection(HailoBBox(0.6, 0.6, 0.2, 0.2), "person", 0.89);
    objects.push_back(first_detection);
    objects.push_back(second_detection);

    return objects;
}

// Default filter function
void filter(HailoROIPtr roi)
{
   std::vector<HailoTensorPtr> tensors = roi->get_tensors();

    std::vector<HailoDetection> detections = demo_detection_objects();
    hailo_common::add_detections(roi, detections); //dection을 ptr에 붙임
}