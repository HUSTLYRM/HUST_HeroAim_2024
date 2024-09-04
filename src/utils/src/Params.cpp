//
// Created by zhiyu on 2021/8/20.
//
#include "../include/Params.h"
#include "cmath"
using namespace std;

namespace ly{
    int CameraParam::device_type;
    string CameraParam::sn;
    string CameraParam::picture_path;
    string CameraParam::video_path;
    int CameraParam::exposure_time;
    float CameraParam::gamma;
    double CameraParam::gain;
    double CameraParam::yaw;
    double CameraParam::pitch;
    double CameraParam::roll;
    int CameraParam::camera_type;
    double CameraParam::fx;
    double CameraParam::fy;
    double CameraParam::u0;
    double CameraParam::v0;
    double CameraParam::k1;
    double CameraParam::k2;
    double CameraParam::k3;
    double CameraParam::p1;
    double CameraParam::p2;
    double CameraParam::camera_trans_x;
    double CameraParam::camera_trans_y;
    double CameraParam::camera_trans_z;
    double CameraParam::height;
    double CameraParam::width;

    string DetectorParam::color;
    int DetectorParam::thresh;

    string SerialParam::device_name;
    SerialPortData SerialParam::recv_data;
    SerialPortWriteData SerialParam::send_data;
    vector<SerialPortData> SerialParam::serial_data_sets(1000);
    int SerialParam::set_id = 0;
    
    bool GlobalParam::DEBUG_MODE;
    int GlobalParam::COLOR;
    int GlobalParam::BIGID;
    bool GlobalParam::SAVE_VIDEO;
    bool GlobalParam::SAVE_ARMOR;
    int GlobalParam::save_step;
    bool GlobalParam::SHOW_THRESH;
    bool GlobalParam::SHOW_COORD;
    bool GlobalParam::SOCKET;
    double GlobalParam::SHOOT_SPEED;
    int GlobalParam::MODE;

    float FilterParams::measurement_noise_pose_x; //测量噪声
    float FilterParams::measurement_noise_pose_y; //测量噪声
    float FilterParams::measurement_noise_pose_z; //测量噪声

    float FilterParams::process_noise_pose_x;
    float FilterParams::process_noise_pose_y;
    float FilterParams::process_noise_pose_z;

    float FilterParams::process_noise_q4_w;
    float FilterParams::process_noise_q4_x;
    float FilterParams::process_noise_q4_y;
    float FilterParams::process_noise_q4_z;

    float FilterParams::stf_beta;
    bool FilterParams::is_use_stf;

    bool FilterParams::is_use_ca_model;

    bool FilterParams::is_use_singer;
    float FilterParams::alpha;
    float FilterParams::max_a_x;
    float FilterParams::max_a_y;
    float FilterParams::max_a_z;

    // 前哨站模式调整的三个参数
    double OutpostParam::time_bias;
    double OutpostParam::time_bias_inverse;
    double OutpostParam::center_ratio;
    double OutpostParam::pitch_bias;

    STATE StateParam::state;

    std::chrono::steady_clock::time_point TimeSystem::time_zero;
}