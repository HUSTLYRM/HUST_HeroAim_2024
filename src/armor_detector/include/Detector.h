//
// Created by zhiyu on 2021/8/20.
//

#ifndef AUTOAIM_DETECTOR_H
#define AUTOAIM_DETECTOR_H

#include "opencv2/opencv.hpp"
#include <algorithm>
#include <math.h>
#include "Log.h"
#include "Params.h"
#include "LightBarFinder.h"
#include "ArmorFinder.h"
#include "../../driver/include/SerialPort.h"
#include "../../driver/include/VideoCapture.h"
#include "Score.h"
#include "Thread.h"
#include "Classifier.h"
#include "../../pose_estimate/include/PoseSolver.h"
#include "../../pose_estimate/include/Predictor.h"
#include <opencv2/dnn.hpp>
#include "ROI_Accelerator.h"
#include "TargetChooser.h"
#include "NumberClassifier.h"
#include "GreenLightDetector.hpp"
//#include "YOLOv10Detector.hpp"


// #include "Bumper.hpp"

// #include <stdlib.h>

using namespace cv;

namespace ly{
    static Ptr<cv::Tracker> tracker;
//    const Scalar hsv_blue_low_boundary = Scalar (100, 43, 46);
//    const Scalar hsv_blue_high_boundary = Scalar (124, 255, 255);

    struct Params_ToDetector{
        Image** frame_pp;
        SerialPortData* SerialPortData_; //传出的串口数据
        // std::chrono::steady_clock::time_point* time_stamp;
        // Mat* armor;
        bool is_update;

        Params_ToDetector(){
//            id = new uint8_t;
            frame_pp = nullptr;
            // armor = new Mat();
            // cache_idx = new uint8_t ;
	        SerialPortData_ = new SerialPortData();
            is_update = false;
        }
    };

    class Detector {
    public:
        explicit Detector();
        ~Detector() = default;
        void setParams(const Params_ToVideo &params_to_video, const Params_ToSerialPort &params_to_serial_port);
        void startDetect(const Params_ToDetector &params, SerialPort* SerialPort_);
        void outpostMode();

    private:
        inline void calcGammaTable(float gamma);
        inline void drawArmorCorners(Mat& drawing, const ArmorBlob& armor, const Scalar& color);
        double fitTrajectory(const Point3d& trans, double v);
        NumberClassifier classifier = NumberClassifier(0);
        // SerialPortWriteData sendData_;
        unordered_map<int, float> gamma_table;
        Params_ToDetector _detector_thread_params;
        // pthread_t threadID{};
        // clock_t pic_time;

        ROIAccelerator *roi_accelerator;
        TargetChooser *target_chooser;

        // TODO 增加一个udp发送，用于调试
        UDPSender * udpsender;
        PoseDataFrame poseDate;
        DeltatFrame delta_t_frame;

    };
}


#endif //AUTOAIM_DETECTOR_H
