//
// Created by MicDZ on 24-7-13.
//

#ifndef AUTOAIM_GREENLIGHTDETECTOR_HPP
#define AUTOAIM_GREENLIGHTDETECTOR_HPP
//
// Created by MicDZ on 2024/4/7.
//

#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
struct bbox {
    float x1;
    float y1;
    float x2;
    float y2;
    int class_id;
    float score;
};

class GreenLightDetector {
public:
    GreenLightDetector();
    GreenLightDetector(string model_path);
    ~GreenLightDetector();
    void detect(const Mat img, vector<Rect2f> &rois, Mat &debugImg, double conf_thresh = 0.45);
    Rect2f getROI(cv::Mat img, bbox result);
    void visualizeResult(const cv::Mat &img, bbox result);
private:
    ///////////// 参数 //////////////
    const double IMG_SIZE = 384;
    std::string model_path = "./model/best_int8_openvino_model/best.xml";
//    float score_threshold = 0.25;
    float nms_threshold = 0.45;

    double p_width = 1.0; // 将模型得到的ROI的长宽适当扩大 p 倍
    double p_height = 1.0;
    ////////////////////////////////

    ov::Core ie;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;


    ov::Tensor input_tensor;
    ov::Tensor output_tensor;
    ov::Shape input_shape;
    ov::Shape output_shape;



    const std::vector<std::string> class_names = {
            "blue", "red" };

    Mat letterbox(const cv::Mat& source);

};


#endif //AUTOAIM_GREENLIGHTDETECTOR_HPP
