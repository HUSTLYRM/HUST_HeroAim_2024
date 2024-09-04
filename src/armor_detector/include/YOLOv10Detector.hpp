//
// Created by MicDZ on 24-7-15.
//

#ifndef YOLOV10_YOLOV10DETECTOR_HPP
#define YOLOV10_YOLOV10DETECTOR_HPP


#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

using namespace cv;
using namespace std;
struct Detection {
    short class_id;
    float confidence;
    cv::Rect2f box;
};


class YOLOv10Detector {
public:


    YOLOv10Detector() {}
    YOLOv10Detector(const std::string &model_path, const float &model_confidence_threshold);
    YOLOv10Detector(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold);

        std::vector<Rect2f> RunInference(const cv::Mat &frame);

    private:
        void InitialModel(const std::string &model_path);
        void Preprocessing(const cv::Mat &frame);
        void PostProcessing();
        cv::Rect2f GetBoundingBox(const cv::Rect2f &src) const;

        cv::Point2f scale_factor_;
        cv::Size2f model_input_shape_;
        cv::Size model_output_shape_;

        ov::InferRequest inference_request_;
        ov::CompiledModel compiled_model_;

        std::vector<Detection> detections_;
        vector<Rect2f> rois_;
        float model_confidence_threshold_;

};
#endif //YOLOV10_YOLOV10DETECTOR_HPP
