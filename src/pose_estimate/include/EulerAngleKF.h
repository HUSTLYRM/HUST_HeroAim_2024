#ifndef _EULER_ANGLE_KF_HPP
#define _EULER_ANGLE_KF_HPP
#include "eigen3/Eigen/Dense"
#include <eigen3/Eigen/Core>
#include <vector>
#include "Config.h"
#include "Log.h"
namespace ly
{
    class EulerAngleKF
    {
        public:
            EulerAngleKF();
            ~EulerAngleKF();
            Eigen::Matrix<double, 8, 1> runKalman(const Eigen::Quaterniond &new_q4, const double &delta_t);
            void setMeasurementNoise(double w_noise,double x_noise,double y_noise,double z_noise);
            bool dont_believe_measure;
            bool if_first_jump=true;
            Eigen::Matrix<double, 8, 1> prior_state_estimate;//先验估计（8*1）
        private:
            double update_time;
            bool is_kalman_init;
            double reproject_error_ratio_threshold;
            
            Eigen::Matrix<double, 8, 4> kalman_gain;//卡尔曼增益
            
            //Eigen::Matrix<double, 8, 1> prior_state_estimate;//先验估计（8*1）
            Eigen::Matrix<double, 4, 1> prior_state_estimate_measure;//先验估计测量（4*1）
            Eigen::Vector4d posteriori_d_Q4;//后验角速度
            Eigen::Vector4d posteriori_Q4;//后验四元数
            Eigen::Matrix<double, 4, 1> residual; //残差向量（4*1）

            Eigen::Matrix<double, 8, 8> transition_matrix; //状态转移矩阵
            Eigen::Matrix<double, 8, 1> posteriori_state_estimate;//后验估计
            Eigen::Matrix<double, 8, 8> error_cov_post;//状态估计协方差矩阵
            Eigen::Matrix<double, 8, 8> process_noise_cov;
            Eigen::Matrix4d process_noice;//过程噪声
            Eigen::Matrix<double, 8, 4> process_noise_matrix;//过程噪声矩阵
            Eigen::Matrix<double, 4, 8> measurement_matrix;//测量矩阵
            Eigen::Matrix<double, 4, 4> measurement_noise_cov;//测量噪声矩阵
            Eigen::Matrix<double, 4, 4> measurement_cov_maxtrix;  //测量协方差

            void rebootKalman(const Eigen::Quaterniond &new_q4);
            void setTransitionMatrix();
            void setUpdateTime(const double &delta_t);
            void setProcessNoise();
            void setMeasurementNoise();
            Eigen::Matrix<double, 8, 1> predict(const Eigen::Quaterniond &new_q4);
            Eigen::Matrix<double, 8, 1> correct(const Eigen::Quaterniond &new_q4);
            Eigen::Matrix<double, 8, 1> update();
    };
}
#endif