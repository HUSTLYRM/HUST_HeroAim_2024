//
// Created by zhiyu on 2021/8/20.
//

#include "../include/Config.h"

using namespace ly;
using namespace std;
using namespace Json;

Config::Config(const string &path)
{
    this->json_file_path = path;
}

void Config::parse()
{
    fstream json_file(json_file_path, std::ios::in);
    LOG_IF(ERROR, !json_file.is_open()) << "can't find json file in " << json_file_path;
    LOG_IF(INFO, json_file.is_open()) << "successfully open json file: " << json_file_path;

    JSONCPP_STRING errs;
    CharReaderBuilder builder;
    Value root;
    bool status = Json::parseFromStream(builder, json_file, &root, &errs);
    LOG_IF(ERROR, !status) << "json file parse error!";
    /*** camera param ***/
    CameraParam::device_type = root["camera"]["device_type"].asInt();
    CameraParam::sn = root["camera"]["sn"].asString();
    CameraParam::video_path = root["camera"]["video_path"].asString();
    CameraParam::picture_path = root["camera"]["picture_path"].asString();
    CameraParam::camera_type = root["camera"]["camera_type"].asInt();
    DLOG(INFO) << "camera type: " << CameraParam::camera_type << "mm";
    if(CameraParam::camera_type == 8){
        CameraParam::exposure_time = root["camera"]["camera_8mm_param"]["exposure_time"].asInt();
        CameraParam::gain = root["camera"]["camera_8mm_param"]["gain"].asDouble();
        CameraParam::gamma = root["camera"]["camera_8mm_param"]["gamma"].asFloat();
        CameraParam::fx = root["camera"]["camera_8mm_param"]["fx"].asDouble();
        CameraParam::fy = root["camera"]["camera_8mm_param"]["fy"].asDouble();
        CameraParam::u0 = root["camera"]["camera_8mm_param"]["u0"].asDouble();
        CameraParam::v0 = root["camera"]["camera_8mm_param"]["v0"].asDouble();
        CameraParam::k1 = root["camera"]["camera_8mm_param"]["k1"].asDouble();
        CameraParam::k2 = root["camera"]["camera_8mm_param"]["k2"].asDouble();
        CameraParam::k3 = root["camera"]["camera_8mm_param"]["k3"].asDouble();
        CameraParam::p1 = root["camera"]["camera_8mm_param"]["p1"].asDouble();
        CameraParam::p2 = root["camera"]["camera_8mm_param"]["p2"].asDouble();
        CameraParam::yaw = root["camera"]["camera_8mm_param"]["yaw"].asDouble();
        CameraParam::pitch = root["camera"]["camera_8mm_param"]["pitch"].asDouble();
        CameraParam::roll = root["camera"]["camera_8mm_param"]["roll"].asDouble();
    } else{
        CameraParam::exposure_time = root["camera"]["camera_param"]["exposure_time"].asInt();
        CameraParam::gain = root["camera"]["camera_param"]["gain"].asDouble();
        CameraParam::gamma = root["camera"]["camera_param"]["gamma"].asFloat();
        CameraParam::fx = root["camera"]["camera_param"]["fx"].asDouble();
        CameraParam::fy = root["camera"]["camera_param"]["fy"].asDouble();
        CameraParam::u0 = root["camera"]["camera_param"]["u0"].asDouble();
        CameraParam::v0 = root["camera"]["camera_param"]["v0"].asDouble();
        CameraParam::k1 = root["camera"]["camera_param"]["k1"].asDouble();
        CameraParam::k2 = root["camera"]["camera_param"]["k2"].asDouble();
        CameraParam::k3 = root["camera"]["camera_param"]["k3"].asDouble();
        CameraParam::p1 = root["camera"]["camera_param"]["p1"].asDouble();
        CameraParam::p2 = root["camera"]["camera_param"]["p2"].asDouble();
        CameraParam::yaw = root["camera"]["camera_param"]["yaw"].asDouble();
        CameraParam::pitch = root["camera"]["camera_param"]["pitch"].asDouble();
        CameraParam::roll = root["camera"]["camera_param"]["roll"].asDouble();
    }
    
    CameraParam::camera_trans_x = root["camera"]["camera_trans"]["x"].asDouble();
    CameraParam::camera_trans_y = root["camera"]["camera_trans"]["y"].asDouble();
    CameraParam::camera_trans_z = root["camera"]["camera_trans"]["z"].asDouble();


    FilterParams::measurement_noise_pose_x = root["detector"]["filter_params"]["measurement_noise_pose_x"].asFloat();
    FilterParams::measurement_noise_pose_y = root["detector"]["filter_params"]["measurement_noise_pose_y"].asFloat();
    FilterParams::measurement_noise_pose_z = root["detector"]["filter_params"]["measurement_noise_pose_z"].asFloat();


    // 主要调整的过程噪声
    FilterParams::process_noise_pose_x = root["detector"]["filter_params"]["process_noise_pose_x"].asFloat();
    FilterParams::process_noise_pose_y = root["detector"]["filter_params"]["process_noise_pose_y"].asFloat();
    FilterParams::process_noise_pose_z = root["detector"]["filter_params"]["process_noise_pose_z"].asFloat();
    
    // EulerAngleFilterParams
    FilterParams::process_noise_q4_w = root["detector"]["filter_params"]["process_noise_q4_w"].asFloat();
    FilterParams::process_noise_q4_x= root["detector"]["filter_params"]["process_noise_q4_x"].asFloat();
    FilterParams::process_noise_q4_y = root["detector"]["filter_params"]["process_noise_q4_y"].asFloat();
    FilterParams::process_noise_q4_z = root["detector"]["filter_params"]["process_noise_q4_z"].asFloat();
    
    FilterParams::stf_beta = root["detector"]["filter_params"]["stf_beta"].asFloat();
    FilterParams::is_use_stf = root["detector"]["filter_params"]["is_use_stf"].asBool();
    FilterParams::is_use_ca_model = root["detector"]["filter_params"]["is_use_ca_model"].asBool();
    FilterParams::is_use_singer = root["detector"]["filter_params"]["is_use_singer"].asBool();
    FilterParams::alpha = root["detector"]["filter_params"]["alpha"].asFloat();
    FilterParams::max_a_x = root["detector"]["filter_params"]["max_a_x"].asFloat();
    FilterParams::max_a_y = root["detector"]["filter_params"]["max_a_y"].asFloat();
    FilterParams::max_a_z = root["detector"]["filter_params"]["max_a_z"].asFloat();

    OutpostParam::time_bias = root["outpost"]["time_bias"].asDouble();
    OutpostParam::time_bias_inverse = root["outpost"]["time_bias_inverse"].asDouble();
    OutpostParam::center_ratio = root["outpost"]["pixel_center_ratio"].asDouble();
    OutpostParam::pitch_bias = root["outpost"]["pitch_bias"].asDouble();


    DetectorParam::color = root["detector"]["color"].asString();
    DetectorParam::thresh = root["detector"]["thresh"].asInt();

    SerialParam::device_name = root["serialport"]["deviceName"].asString();

    GlobalParam::MODE = root["debug"]["mode"].asInt();
    GlobalParam::COLOR = root["debug"]["color"].asInt();
    GlobalParam::BIGID = root["debug"]["big_id"].asInt();
    GlobalParam::DEBUG_MODE = root["debug"]["enable"].asBool();
    GlobalParam::SAVE_VIDEO = root["debug"]["video"].asBool();
    GlobalParam::SAVE_ARMOR = root["debug"]["armor"].asBool();
    GlobalParam::SHOW_THRESH = root["debug"]["thresh"].asBool();
    GlobalParam::save_step = root["debug"]["save_step"].asInt();
    GlobalParam::SHOW_COORD = root["debug"]["coord"].asBool();
    GlobalParam::SHOOT_SPEED = root["debug"]["shoot_speed"].asDouble();
    GlobalParam::SOCKET = root["debug"]["socket"].asBool();

    json_file.close();
}
