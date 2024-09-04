// //
// // Created by zhiyu on 2021/8/20.
// //

// #ifndef PRIDICTOR_H
// #define PRIDICTOR_H

// #include <unistd.h>
// #include "Config.h"
// #include <utility>
// #include <opencv2/opencv.hpp>
// #include <eigen3/Eigen/Dense>
// #include <opencv2/core/eigen.hpp>
// #include "Params.h"
// #include "../../armor_detector/include/ArmorFinder.h"
// // #include "Kalman.h"
// #include <string>
// #include <arpa/inet.h>
// #include <sys/socket.h>
// #include <netinet/in.h>
// #include "../../utils/include/UDPSender.hpp"
// #include "NormalEKF.h"
// #include <sophus/se3.h>

// using namespace cv;
// using namespace std;

// namespace ly
// {
//     typedef struct
//     {
//         float pitch; // rad
//         float yaw;   // rad
//         float time;  // 击打弹道时间(ms)
//         float distance;
//     } Angle_t;

//     class Predictor
//     {
//     public:
    
//         Predictor();
//         ~Predictor();
//         double fitTrajectory(const Point3d& armor, double v, bool useRoll=false);
//         double getYaw(const Point3d& armor, double delta_t, double v, const SerialPortData& imu_data, bool usePredictor=true);
//         void setStatePost(float x, float z);
//         Mat getCurState();
        
//         Angle_t Predict(const Sophus::SE3 &armor_pose, const Sophus::SE3 &armor_pose_sec, bool is_get_second_armor, int detect_mode, SerialPortData SerialPortData_, float &frame_delta_t);

//         // 根据物理方程进行结算
//         Angle_t ballistic_equation(float gim_pitch, const Eigen::Vector3d &armor_Position);
//         float BulletModel(float x, float v, float angle);

//         // 重启卡尔曼
//         void reset();

//     private:
//         double fitPNP(const Point3d& armor, bool usePredictor=true);

//         UDPSender * udpsender;
//         AimKFFrame aimKFframe;

//         NormalEKF *ekf_filter;
//         Eigen::Vector3d last_pose_vec;
//         float ShootSpeed;
//         Angle_t shootAngleTime_now;
//         Angle_t shootAngleTime_pre;

//         AimFrame data;


//         KalmanFilter position_predictor;
//         Mat pos_m;
//         bool init = false;

//         KalmanFilter sentinel_predictor;
//         Mat sentinel_m;

//         float distance;
//         float shoot_t;
//         Mat position_cur;

        

//         // sockaddr_in server_addr;
//         // int client_socket;
//         // string buffer;
//         // char write_str[40];

//         float last_x;
//         float last_z;
//         float last_a;
//         float last_xv;
//         float last_zv;
//     };

// }

// #endif //AUTOAIM_POSESOLVER_H
