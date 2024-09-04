// /**
//  * dependence:opencv,eigen,sophus
//  * usage:
//  *      use runKalman(const ArmorPose &new_armor_pose, const float &delta_t)
//  *      input the location vector (x,y,z) relative to camera , and the delta_t
//  *      the function will return the vector predicted(corrected)
//  * 
//  *      if the speed or pose of armor need to be used after call runKalman()
//  *          use getSpeed() to get the predict speed
//  *          use getPose() to get the predict armor pose
//  * 
//  *      if the target has lost(this should be judged outside),use resetKalman() to reset Kalman Filter
//  *
// **/
// #ifndef _KALMAN_H
// #define _KALMAN_H

// #include <opencv2/opencv.hpp>
// #include <eigen3/Eigen/Dense>
// #include <opencv2/core/eigen.hpp>

// namespace ly
// {
//     // x,y,z pnp 解算的位姿
//     typedef Eigen::Vector3d ArmorPose;  //x,y,z translation vector supported by pnp
//     // xv,yv,zx 装甲板运动的速度
//     typedef Eigen::Vector3d ArmorSpeed; //x_v,y_v,z_v
//     class Kalman
//     {
//         struct ArmorState
//         {
//             ArmorPose pose;
//             ArmorSpeed speed;
//         };
//     public:
//         cv::KalmanFilter KF_;   // 内部使用opencv提供的一个KalmanFilter对象
//     private:
//         cv::Mat measurement_;   // 测量矩阵 H 

//         bool is_kalman_init = false; //kalman init symbol
//         bool is_second_find = false;    // 第二次找到 计算speed
//         bool is_continuous_find = false; //continously find armor 持续找到装甲板

//         void resetTransitionMatrix();
//         void setTransitionMatrix(float delta_t); //ms
//         void updateArmorState(const ArmorPose &new_armor_pose);
//         void setUpdateTime(const float &delta_t);
//         void calculateSpeed(const ArmorPose &new_armor_pose);
//         void updateMeasurement(const ArmorPose &new_armor_pose);
//         void rebootKalman(const ArmorPose &new_armor_pose);
//         ArmorPose correct();    // 更新
//         ArmorPose predict();    // 预测

//         ArmorState last_armor_state;        // 上一帧的预测 
//         ArmorState this_armor_pre_estimate; // 这一帧的预测
//         ArmorSpeed this_armor_speed;        //the newest armor speed estimate 这一帧的速度
//         float update_time;                  // 更新的时间

//     public:
//         Kalman(/* args */);
//         ArmorPose runKalman(const ArmorPose &new_armor_pose, const float &delta_t);
//         ArmorSpeed getSpeed() { return last_armor_state.speed; };
//         ArmorPose getPose() { return last_armor_state.pose; };
//         void resetKalman();
//         ~Kalman();
//     };
// }

// #endif
