#ifndef _NORMAN_EKF_H
#define _NORMAN_EKF_H
#include "Config.h"
#include "Log.h"
#include <chrono>
#include "ExtendedKalman.hpp"

#define VERIFY_THRESH 10
namespace ly
{
    // 将xyz转成pithc、yaw、distance
    class Xyz2Pyd
    {
    public:
        template <class T>
        // x,x_v, y,y_v, z,z_v
        void operator()(const T xyz[6], T pyd[3]) // ms
        {
            pyd[0] = ceres::atan2(xyz[4], ceres::sqrt(xyz[0] * xyz[0] + xyz[2] * xyz[2])); // pitch = z / sqrt(x^2+y^2)
            // yaw = atan(-x/y)
            pyd[1] = ceres::atan2(-xyz[0], xyz[2]);                                        // yaw = -x / y
            pyd[2] = ceres::sqrt(xyz[0] * xyz[0] + xyz[4] * xyz[4] + xyz[2] * xyz[2]);     // distance
        }
    };
    class NormalEKF
    {
    private:
        /* data */
        // 扩展卡尔曼滤波
        ExtendedKalman<double, 6, 3> *kalman_filter; // 滤波6维，x,x_v, y,y_v, z,z_v ,测量三维x,y,z
        bool is_kalman_init;
        void rebootKalman(const Eigen::Vector3d &new_armor_pose);
        void resetTransitionMatrix();
        void setUpdateTime(const double &delta_t);
        void setTransitionMatrix();
        Eigen::Vector3d correct(const Eigen::Vector3d &armor_pose);
        void setMeasureMatrix();
        void setMeasurementNoise(const Eigen::Vector3d &armor_pose);
        void setProcessNoise();
        void rebootKalmanWithInitial(const Eigen::Vector3d &new_armor_pose);

        double update_time;
        Eigen::Vector3d posteriori_speed;
        Eigen::Vector3d posteriori_pose;

        Eigen::Matrix3d process_noice;
        Eigen::Matrix<double, 6, 3> process_noise_matrix;

        Eigen::Vector3d measure(const Eigen::Vector3d &armor_pose);
        void setIsUseSTF(bool flag);

        Eigen::Vector3d pyd;

        Xyz2Pyd xyz_to_pyd;

    public:
        NormalEKF(/* args */);
        ~NormalEKF();
        Eigen::Vector3d runKalman(const Eigen::Vector3d &new_armor_pose, const double &delta_t);
        Eigen::Vector3d predict(const double &predict_t);
        Eigen::Vector3d getSpeed() { return posteriori_speed; };
        Eigen::Vector3d getPose() { return posteriori_pose; };
        Eigen::Vector3d getPYD() { return pyd; }

        void resetKalman()
        {
            is_kalman_init = false;
        }
        Eigen::Vector3d getResidual() { return kalman_filter->residual; };
        double getDetectParam() { return detect_param; };
        void setProcessNoise(double x, double y, double z);

        double detect_param = 0;
    };

}
#endif