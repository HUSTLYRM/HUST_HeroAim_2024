//
// Created by zhiyu on 2021/8/20.
//

#include "../include/VideoCapture.h"
#include "SerialPort.h"

using namespace ly;
using namespace cv;

ly::VideoCapture::VideoCapture() {
    width = 1280;
    height = 1024;
    offset_x = 0;
    offset_y = 0;
}

ly::VideoCapture::~VideoCapture() {

}

void ly::VideoCapture::chooseCameraType(ly::VideoCapture *& video) {
    switch (CameraParam::device_type) {
        case DaHen:                         // 0: 用大恒相机
            video = new DaHenCamera();
            break;
        case Video:                         // 1: 用本地视频
            video = new NativeVideo();
            break;
        case Picture:                       // 2: 用本地图片
            video = new NativePicture();
            break;
        default:
            video = new NativeVideo();
            break;
    }
    video->open();
}