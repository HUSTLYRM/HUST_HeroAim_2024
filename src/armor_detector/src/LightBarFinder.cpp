//
// Created by zhiyu on 2021/8/24.
//

#include "LightBarFinder.h"

using namespace ly;
using namespace cv;

bool LightBarFinder::findLightBarBlobs(const Mat &frame, LightBarBlobs &lightBarBlobs) {
    Mat mat, hsv, green, white;
    vector<Mat> channels;
    vector<vector<Point>> contours;
    split(frame, channels);
    // 白光补偿
    // inRange(frame, Scalar(150, 150, 150), Scalar(255, 255, 255), white);

    // 101: 自己是蓝色
    SerialParam::recv_data.color = 1;   // 写明自己是红色
    SerialParam::recv_data.color = 101;

    // if(DetectorParam::color == "red"){
    if(SerialParam::recv_data.color == 101){
        // Mat lower, upper;
        // cvtColor(frame, hsv, COLOR_BGR2HSV);
        // inRange(hsv, Scalar(0, 160, 102), Scalar(20, 255, 255), lower);
        // inRange(hsv, Scalar(160, 160, 102), Scalar(180, 255, 255), upper);
        // bitwise_or(lower, upper, mat);
        subtract(channels[2], channels[0], mat);        // BGR R-B  突出红色
        // subtract(mat, channels[1], mat);
    } else{
        subtract(channels[0], channels[2], mat);        // B-R 突出蓝色
        // subtract(mat, channels[1], mat);
    }
    
    threshold(mat, mat, DetectorParam::thresh, 255, THRESH_BINARY);
    // extract green
    subtract(channels[1], channels[2], green);
    subtract(green, channels[0], green);
    threshold(green, green, 100, 255, THRESH_BINARY_INV);
    // merge
    bitwise_and(mat, green, mat);
    // bitwise_or(mat, white, mat);
    morphologyEx(mat, mat, MORPH_CLOSE, kernel1);
    // morphologyEx(mat, mat, MORPH_OPEN, kernel2);
    
    if(GlobalParam::SHOW_THRESH){
        imshow("thresh", mat);
        // imshow("white", white);
        waitKey(1);
    }
    
    findContours(mat, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for(const auto& contour: contours){
        const RotatedRect& rrect = minAreaRect(contour);
        if(isValidLightBarBlob(rrect)){
            lightBarBlobs.emplace_back(rrect);
        }
    }

    sort(lightBarBlobs.begin(), lightBarBlobs.end(), [](const RotatedRect& a, const RotatedRect& b)->bool {
        if(a.center.x != b.center.x) return a.center.x < b.center.x;
        return a.center.y > b.center.y;
    });
    return lightBarBlobs.size() >= 2;
}

bool LightBarFinder::isValidLightBarBlob(const RotatedRect& rrect){
    if(
            checkAspectRatio(rrect.size.aspectRatio()) &&
            checkArea(rrect.size.area()) && 
            checkAngle(rrect)
            ){
        return true;
    }
    // DLOG(INFO) << "not lightbar: " << rrect.size.aspectRatio() << " " << rrect.size.area() << " " << rrect.angle;
    return false;
}

bool LightBarFinder::checkAspectRatio(double ratio) {
    // return true;
    return ratio <= 10 && ratio >= 2.5/2 || ratio <= 2./2.5 && ratio >= 1./10;
    // return ratio <= 10 && ratio >= 1./10;
}

bool LightBarFinder::checkArea(double area) {
    return area >= 20 && area < 400*100;
}

bool LightBarFinder::checkAngle(const RotatedRect& rrect) {
    double angle = rrect.angle;
    return rrect.size.width < rrect.size.height ? angle <= 30 : angle >= 60;
}

LightBarFinder::LightBarFinder(){
    kernel1 = getStructuringElement(0, Size(5, 5));
    // kernel1 = getStructuringElement(0, Size(11, 11));
    kernel2 = getStructuringElement(0, Size(5, 5));
}

