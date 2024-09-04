//
// Created by narrow on 2022/4/24.
//

#include "Classifier.h"

using namespace ly;

CNN::CNN() {
    net = dnn::readNetFromONNX(model);
}

int CNN::predict(const Mat &frame, float& prob, float prob_thresh) {
    Mat blob;
    dnn::blobFromImage(frame, blob, 1, Size(32, 32), Scalar(0, 0, 0), false, false, CV_32FC1);
    net.setInput(blob);
    Mat predict = net.forward();
    softmax(predict);
    int class_ = 0;
    for(int i=0;i<9;i++){ 
        if(predict.at<float>(i) > prob){
            prob = predict.at<float>(i);
            class_ = i;
        }
    }
    // DLOG(INFO) << " class: " << class_ << " prob: " << prob << endl;
    return prob > prob_thresh ? class_ : 0;
    // return class_;
}

int CNN::predict(const Mat &frame) {
    Mat blob;
    dnn::blobFromImage(frame, blob, 1, Size(32, 32), Scalar(0, 0, 0), false, false, CV_32FC1);
    net.setInput(blob);
    Mat predict = net.forward();
    softmax(predict);
    int class_ = 0;
    float prob = 0;
    for(int i=0;i<9;i++){ 
        if(predict.at<float>(i) > prob){
            prob = predict.at<float>(i);
            class_ = i;
        }
    }
    // DLOG(INFO) << " class: " << class_ << " prob: " << prob << endl;
    return prob > 0.85 ? class_ : 0;
    // return class_;
}

void CNN::train() {

}

void CNN::softmax(Mat& mat) {
    float denominator = 0;
    for(int i=0;i<9;i++){
        denominator += exp(mat.at<float>(i));
    }
    for(int i=0;i<9;i++){
        mat.at<float>(i) = exp(mat.at<float>(i)) / denominator;
    }
}
