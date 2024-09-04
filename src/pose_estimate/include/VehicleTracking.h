// #ifndef _VEHICEL_TRACKING_H
// #define _VEHICEL_TRACKING_H
// #include <eigen3/Eigen/Dense>
// #include <Eigen/SVD>
// // #include <opencv2/viz.hpp>
// #include "sophus/se3.h"
// #include "sophus/so3.h"
// #include "ExtendedKalman.hpp"
// #include <vector>
// #include "Log.h"
// namespace ly
// {
//     class VehicleTracking
//     {

//     public:
//         VehicleTracking();
//         ~VehicleTracking();
//         Eigen::Vector3d predictVehicleState(const Sophus::SE3 &armor_pose, const Sophus::SE3 &armor_pose_sec, bool is_get_second_armor, int &detect_mode, float shoot_time, float frame_delta_t, float yaw_gimbal);
//         void resetTracker();
//         // void vizDebug();
//         std::vector<Eigen::Vector4d> getArmorSerial() { return armor_serial_base; };
//         float getYaw() { return yaw_send; };
//         std::vector<float> speed_vector = std::vector<float>(3);

//     private:
//         ExtendedKalman<double, 9, 16> *extended_kalman_filter; // 状态向量9维，x_c,y_c,z_c,yaw,v_xc,v_yc,v_zc,v_yaw,r ,测量十六维x_1,y_1,z_1,yaw_1,x_2,y_2,z_2,yaw_2,x_3,y_3,z_3,yaw_3,x_4,y_4,z_4,yaw_4
//         void setUpdateTime(const double &delta_t);

//         Eigen::Vector3d getPredictPoint(const Eigen::Matrix<double, 9, 1> whole_car_state, float &shoot_time);

//         Eigen::Vector4d PoseToMeasurement(const Sophus::SE3 &pose);

//         // 设置观测向量,x_1,y_1,z_1,yaw_1,x_2,y_2,z_2,yaw_2,x_3,y_3,z_3,yaw_3,x_4,y_4,z_4,yaw_4
//         void setMeasurementVector();

//         void setQandRMatrix();

//         // 生成装甲板序列基准
//         void setArmorSerialBase(Eigen::Vector4d armor_measure, Eigen::Vector4d armor_measure_sec, bool is_get_second_armor);

//         // 更新装甲板序列基准
//         void updateArmorSerialBase();

//         // 通过与序列基准对比确定目前的装甲板序号
//         Eigen::Matrix<double, 16, 1> ComfirmArmorSerialNow(Eigen::Vector4d &armor_measure, Eigen::Vector4d &armor_measure_sec, bool is_get_second_armor);

//         // 无量测更新kalman
//         Eigen::Matrix<double, 9, 1> runKalmanWithoutMeasure();

//         // 正常有量测更新kalman
//         Eigen::Matrix<double, 9, 1> runKalmanWithMeasure(Eigen::Matrix<double, 16, 1> &measure);

//         Eigen::Matrix<double, 16, 1> makeMeasureVector(Eigen::Vector4d &armor_measure, int &cur_armor_index);
//         Eigen::Matrix<double, 16, 1> makeMeasureVector(Eigen::Vector4d &armor_measure, Eigen::Vector4d &armor_measure_sec, int &cur_left_armor_index);

//         void rebootKalman();

//         void setTransitionMatrix();
//         void setObservationMatrix();

//         double update_time;
//         bool is_kalman_init;
//         std::vector<Eigen::Vector4d> armor_serial_base = std::vector<Eigen::Vector4d>(4);

//         // 状态向量9维，x_c,y_c,z_c,yaw,v_xc,v_yc,v_zc,v_yaw,r
//         Eigen::Matrix<double, 9, 1> vehicle_state;

//         float yaw_send;

//         // cv::viz::Viz3d window;
//         // cv::viz::WCoordinateSystem coordinate_system;
//     };

// }
// #endif

#ifndef _VEHICEL_TRACKING_H
#define _VEHICEL_TRACKING_H
#include <eigen3/Eigen/Dense>
#include <Eigen/SVD>
// #include <opencv2/viz.hpp>
#include "sophus/se3.h"
#include "sophus/so3.h"
#include "ExtendedKalman.hpp"
#include <vector>
#include "Log.h"
#include "Params.h"
namespace ly
{
    class VehicleTracking
    {

    public:
        VehicleTracking();
        ~VehicleTracking();
        Eigen::Vector3d predictVehicleState(const Sophus::SE3 &armor_pose, const Sophus::SE3 &armor_pose_sec, bool is_get_second_armor, int &detect_mode, float shoot_time, float frame_delta_t, float yaw_gimbal);
        void resetTracker();
        std::vector<Eigen::Vector3d> getArmorSerial() { return armor_serial; }
        float getYaw() { return yaw_send; };
        Eigen::Matrix<double, 9, 1> getVehicleState() { return vehicle_state; }
        std::vector<float> speed_vector = std::vector<float>(3);
        bool armor_switch = false;

    private:
        ExtendedKalman<double, 9, 4> *extended_kalman_filter; // 状态向量9维，x_c,v_x,y_c,v_y,z_c,v_z,yaw,v_yaw,r ,测量四维x,y,z,yaw
        void setUpdateTime(const double &delta_t);

        Eigen::Vector3d getPredictPoint(const Eigen::Matrix<double, 9, 1> whole_car_state, float &shoot_time, float yaw_gimbal);

        Eigen::Vector4d PoseToMeasurement(const Sophus::SE3 &pose);

        // 设置观测向量,x_1,y_1,z_1,yaw_1
        void setMeasurementVector();

        void setQandRMatrix();

        void getVehicleState(Eigen::Vector4d &measure);

        Eigen::Vector3d getArmorPositionFromState(Eigen::Matrix<double, 9, 1>);

        void handleArmorJump(Eigen::Vector4d &measure);

        void rebootKalman(bool, Eigen::Vector4d, Eigen::Vector4d);

        void setTransitionMatrix();
        void setObservationMatrix();

        double update_time;
        bool is_kalman_init;

        // 状态向量9维，x_c,v_x,y_c,v_y,z_c,v_z,yaw,v_yaw,r
        Eigen::Matrix<double, 9, 1> vehicle_state;

        std::vector<Eigen::Vector3d> armor_serial;

        float yaw_send;

        double last_z, last_r, last_yaw;
    };

}
#endif