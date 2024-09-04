//
// Created by zhiyu on 2021/8/20.
//

#ifndef AUTOAIM_POSESOLVER_H
#define AUTOAIM_POSESOLVER_H
#include <fstream>
#include <sstream>
#include "Config.h"
#include "Predictor_main.h"
#include "../../armor_detector/include/ArmorFinder.h"
#include "../../utils/include/UDPSender.hpp"
#include "Params.h"
#include "Array.hpp"
#include "newArray.hpp"

#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.h>
#include <sophus/so3.h>
#include <fstream>
#include <string>
#include <chrono>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "Bumper.hpp"
#include <set>
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace Sophus;

namespace ly
{
    struct GimbalPose {
        double pitch;
        double yaw;
        double roll;
    };
    class PoseSolver
    {
    public:
        explicit PoseSolver();
        // 更新辅瞄
        Point2f antiTop(vector<ArmorBlob> &armors, double delta_t, const SerialPortData& imu_data, SerialPort* SerialPort_, bool& getTarget);
        bool getPoseInCamera(vector<ArmorBlob> &armors, double delta_t, const SerialPortData imu_data, SerialPort* SerialPort_, int &this_frame_class, int &last_frame_class);

        Point2f outpostMode(vector<ArmorBlob> &armors, double delta_t, const SerialPortData& imu_data, SerialPort* SerialPort_, bool& getTarget);
        Point2f greenShoot(vector<Rect2f> green, double delta_t, const SerialPortData &imu_data, SerialPort *SerialPort_,
                   bool &getCenter);
        bool adjustToPoint(Point2f greenTarget, Point2f cameraTarget, const SerialPortData &imu_data,
                                       int& modeNum,bool quickStart=false,bool restart=false);
        Point2f halfoutpostMode(vector<ArmorBlob> &armors, double delta_t, const SerialPortData& imu_data, SerialPort* SerialPort_);
        void sentinelMode(vector<ArmorBlob> &armors, double delta_t, const SerialPortData& imu_data, SerialPort* SerialPort_);
        void clearCircle();
        void clearSentinel();
        Point2f reproject(Point3f center);  // 将中心进行重投影
        void update_delta_t(double &delta_time);
    private:


        void setCameraMatrix(double fx, double fy, double u0, double v0);
        void setDistortionCoefficients(double k_1, double k_2, double p_1, double p_2, double k_3);
        void setimu(float pitch, float yaw, float roll);
        //        void setArmorPoints();
        Sophus::SE3 solveArmor(ArmorBlob& armors, const SerialPortData& imu_data);
        void solveArmorBA(ArmorBlob& armors, const SerialPortData& imu_data);
        float calcDiff(const ArmorBlob& a, const ArmorBlob& b);
        float cosineLaw(float a, float b, float c);
        int chooseArmor(const vector<ArmorBlob>& armors);


        float length_of_small = 0.0675f;        // 2倍左右
        float height_of_small = 0.0275f;
        float length_of_big = 0.1125f;          // 4倍左右
        float height_of_big = 0.0275f;
        // 保存数据到指定txt，用一个file
        ofstream file_1, file_2;

        // 小装甲板3d坐标
        vector<Point3f> points_small_3d = {Point3f(-length_of_small, -height_of_small, 0.f),
                                           Point3f(length_of_small, -height_of_small, 0.f),
                                           Point3f(length_of_small, height_of_small, 0.f),
                                           Point3f(-length_of_small, height_of_small, 0.f)};
        // 使用中点的解算
//        vector<Point3f> points_small_3d = {Point3f(0, -height_of_small, 0.f),
//                                           Point3f(length_of_small, 0, 0.f),
//                                           Point3f(0, height_of_small, 0.f),
//                                           Point3f(-length_of_small, 0, 0.f)};
//          大装甲板3d坐标
        vector<Point3f> points_large_3d = {Point3f(-length_of_big, -height_of_big, 0.f),
                                           Point3f(length_of_big, -height_of_big, 0.f),
                                           Point3f(length_of_big, height_of_big, 0.f),
                                           Point3f(-length_of_big, height_of_big, 0.f)};
        //使用中点的解算
//        vector<Point3f> points_large_3d = {Point3f(0, -height_of_big, 0.f),
//                                                Point3f(length_of_big, 0, 0.f),
//                                                Point3f(0, height_of_big, 0.f),
//                                                Point3f(-length_of_big, 0, 0.f)};
        Predictor* predictor;

        // 解算pnp需要的装甲板坐标
        vector<Point3f> points_3d;
        Vec2d reproject_error;
        vector<float> reprojectionError;
        std::vector<Mat> tvecs;
        std::vector<Mat> rvecs;


        // TODO 增加一个udp发送，用于调试
        UDPSender *udpsender = new UDPSender("192.168.1.11", 3000);
        PoseDataFrame outpostPoseDataFrame;
        CenterFrame centerFrame;

        Mat camera_matrix;
        Mat distortion_coefficients;
        Mat tvec;
        Mat rvec;
        Mat m_T;
        Mat m_R;

        Matrix3d e_R;
        Vector3d e_T;

        // armor的yaw方向角度
        double yaw;

        // vector<Point3f> points_large_3d;
        // vector<Point3f> points_small_3d;

        Sophus::SE3 armor_to_camera;
        Sophus::SE3 camera_to_gimbal; // imu's world
        Sophus::SE3 armor_to_gimbal;
        Sophus::SE3 armor_to_world;
        Sophus::SE3 gimbal_to_world;

        Sophus::SE3 camera_to_world; // imu's world

        // Predictor* predictor;

        ArmorBlob last_armor;
        ArmorBlob armor;

        bool right_clicked = true;
        bool last_right_clicked = true;
        bool first = true;
        bool has_same_class = false;
        bool find_outpost = false;
        bool shoot = false;

        int top_pri = 3;

        int top_cnt = 0;
        int lost_cnt = 0;
        std::chrono::time_point<std::chrono::steady_clock> top_begin;
        std::chrono::time_point<std::chrono::steady_clock> top_exit;
        double exit_duration = 0;
        std::chrono::time_point<std::chrono::steady_clock> shoot_begin;
        double shoot_duration = 0;
        std::chrono::time_point<std::chrono::steady_clock> w_begin;
        double w_duration = 0;
        bool w_init = false;

        // double x0, y0, z0, r, angle0;
        Point3d center;
        Point3d cur;
        // double armor_y = 0;

        // 记录
        std::chrono::steady_clock::time_point send_shoot_time;

        // 5ms 每帧，0.2r/s
        // RollingArray<Point3d> circle = RollingArray<Point3d>(100);

        // 获得5次
        RollingArray<Point3d> outpost_center = RollingArray<Point3d>(40);   // 取10次的中心的

        RollingArray<int> filtered_pitch = RollingArray<int>(100);

        // RollingArray<Point3d> outpost_pixel = RollingArray<Point3d>(100);
        // RollingArray<Point3d> outpost = RollingArray<Point3d>(100);

        // 创建outpost
        newRollingArray outpost = newRollingArray(120);     // 100fps

        RollingArray<double> roll = RollingArray<double>(100);
        newIntRollingArray thresh = newIntRollingArray(20);  // 为了找到比较低的哪一个

        RollingArray<Point3d> sentinel = RollingArray<Point3d>(100);

        float delta_t;
        ceres::CostFunction *cost = nullptr;
        ceres::LossFunction* loss = nullptr;
        // antiTop
        // 用一个set存储并排序所有的armor
        vector<ArmorBlob> armors_set;
        std::vector<std::chrono::steady_clock::time_point> time;
        // 用一个vector存储每一个armor到达最近位置的时间
        std::vector<std::chrono::milliseconds> timeInZone;
        // 发弹延迟
        double tmp_time=163; // 由链路决定的发弹延迟

        double time_bias=OutpostParam::time_bias;

        double time_bias_inverse = OutpostParam::time_bias_inverse;
//        double time_bias = 70;
//        double time_bias_inverse = -150;
        const string target_mode_str[4] = {"NOT_GET_TARGET", "CONTINOUS_GET_TARGET", "LOST_BUMP", "DETECT_BUMP"};
        // TODO 在这里写死了弹丸速度
        double speed = GlobalParam::SHOOT_SPEED;
        GimbalPose solveGimbalPose(Point3d shootTarget);


    };
    class DistanceFromCircleCost {
    public:
        DistanceFromCircleCost(double xx, double yy) : xx_(xx), yy_(yy) {}
        template <typename T> bool operator()(const T* const x,
                                              const T* const y,
                                              const T* const m,  // r = m^2
                                              T* residual) const {
            // Since the radius is parameterized as m^2, unpack m to get r.
            T r = *m * *m;
            // Get the position of the sample in the circle's coordinate system.
            T xp = xx_ - *x;
            T yp = yy_ - *y;
            residual[0] = r*r - xp*xp - yp*yp;
            return true;
        }
    private:
        // The measured x,y coordinate that should be on the circle.
        double xx_, yy_;
    };
}

#endif //AUTOAIM_POSESOLVER_H