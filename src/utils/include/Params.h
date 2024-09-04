//
// Created by zhiyu on 2021/8/20.
//

#ifndef AUTOAIM_PARAMS_H
#define AUTOAIM_PARAMS_H

#include <string>
#include "../../driver/include/SerialPort.h"
#include "opencv2/opencv.hpp"

using namespace std;

namespace ly{
    class CameraParam{
    public:
        static int device_type;
        static string sn;
        static string video_path;
        static string picture_path;
        static int exposure_time;
        static double gain;
        static float gamma;
        static double yaw;
        static double pitch;
        static double roll;
        
        static int camera_type;
        static double fx; 
        static double fy;
        static double u0;
        static double v0;
        static double k1;
        static double k2;
        static double k3;
        static double p1;
        static double p2;
        static double camera_trans_x;
        static double camera_trans_y;
        static double camera_trans_z;

        static double height;
        static double width;
    };

    class DetectorParam{
    public:
        static string color;
        static int thresh;


    };

    class FilterParams
    {
    public:
        static float measurement_noise_pose_x; //测量噪声，计算速度的噪声是这个噪声的两倍
        static float measurement_noise_pose_y; //测量噪声，计算速度的噪声是这个噪声的两倍
        static float measurement_noise_pose_z;

        static float process_noise_pose_x;
        static float process_noise_pose_y;
        static float process_noise_pose_z;

        static float process_noise_q4_w;
        static float process_noise_q4_x;
        static float process_noise_q4_y;
        static float process_noise_q4_z;

        static float stf_beta;
        static bool is_use_stf;

        static bool is_use_ca_model;

        //singer模型
        static bool is_use_singer;
        static float alpha;
        static float max_a_x;
        static float max_a_y;
        static float max_a_z;
    };

    class OutpostParam{
    public:
        static double time_bias;
        static double time_bias_inverse;
        static double center_ratio;
        static double pitch_bias;
    };


    class SerialParam{
    public:
        static bool enable;
        static string device_name;
        static SerialPortData recv_data;
        static SerialPortWriteData send_data;
        static vector<SerialPortData> serial_data_sets;
        static int set_id;
    };

    class GlobalParam{
    public:
        static int MODE;
        static int COLOR;
        static int BIGID;
        static bool DEBUG_MODE;
        static bool SAVE_VIDEO;
        static bool SAVE_ARMOR;
        static bool SHOW_THRESH;
        static int save_step;
        static bool SHOW_COORD;
        static double SHOOT_SPEED;
        static bool SOCKET;
    };

    typedef enum {
        AUTOAIM,
        ANTITOP,
        OUTPOST,
        Half_OUTPOST,
        AUTOAIM_WITH_ROI,
        GREEN_SHOOT
    } STATE;

    typedef enum {
        HERO,
        SENTRY,
        ENGINEER,
    } PRI;

    class StateParam{
    public:
        static STATE state;
    };

    class TimeSystem{
    public:
        static std::chrono::steady_clock::time_point time_zero;
    };
}

#endif //AUTOAIM_PARAMS_H
