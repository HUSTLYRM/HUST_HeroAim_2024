#ifndef _EXTENDED_KALMAN_OLD_HPP
#define _EXTENDED_KALMAN_OLD_HPP
#include "ceres/jet.h"
#include "eigen3/Eigen/Dense"
#include <eigen3/Eigen/Core>
#include <vector>


// EKF模板
template <typename T, int x, int y>
class ExtendedKalman
{
#define ChiSquaredTest_EKF_THRESH 10
public:
    Eigen::Matrix<T, x, y> kalman_gain; //卡尔曼增益

    Eigen::Matrix<T, y, 1> measurement; //测量值z_k

    Eigen::Matrix<T, x, 1> prior_state_estimate;         //先验估计
    Eigen::Matrix<T, y, 1> prior_state_estimate_measure; //先验估计测量

    Eigen::Matrix<T, x, x> transition_matrix;         // A:状态转移矩阵
    Eigen::Matrix<T, x, 1> posteriori_state_estimate; //后验估计
    Eigen::Matrix<T, x, x> error_cov_post;            //状态估计协方差矩阵 P_k
    Eigen::Matrix<T, x, x> process_noise_cov;         //过程噪声矩阵：Q
    Eigen::Matrix<T, y, y> measurement_noise_cov;     //测量噪声矩阵 : R
    Eigen::Matrix<T, x, 1> control_vector;            // u_k：控制向量
    Eigen::Matrix<T, x, x> control_matrix;            // B： 控制矩阵
    Eigen::Matrix<T, y, x> measurement_matrix;        //测量矩阵 ： H

    Eigen::Matrix<T, y, y> measurement_cov_maxtrix;

    Eigen::Matrix<T, y, 1> residual; //残差向量

    Eigen::Matrix<T, y, y> residual_matrix; // 残差矩阵S

public:
    ExtendedKalman()
    {
        error_cov_post = Eigen::Matrix<T, x, x>::Identity();
        control_vector = Eigen::Matrix<T, x, 1>::Zero();
        control_matrix = Eigen::Matrix<T, x, x>::Zero();
        measurement_matrix = Eigen::Matrix<T, y, x>::Zero();
        process_noise_cov = Eigen::Matrix<T, x, x>::Identity() * 0.001; //表明对预测过程的相信程度
        measurement_noise_cov = Eigen::Matrix<T, y, y>::Identity();     //证明对测量过程的相信程度
        posteriori_state_estimate = Eigen::Matrix<T, x, 1>::Zero();     //后验估计由外部进行初始化
        measurement_cov_maxtrix = Eigen::Matrix<T, y, y>::Zero();
    }

    template <class Func>
    Eigen::Matrix<T, x, 1> predict(Func &&func, const Eigen::Matrix<T, y, 1> &measure_vec) //根据上一帧的估计值直接利用函数进行计算
    {
        prior_state_estimate = transition_matrix * posteriori_state_estimate + control_matrix * control_vector;

        error_cov_post = transition_matrix * error_cov_post * transition_matrix.transpose() + process_noise_cov;

        ceres::Jet<double, x> Xp_auto_jet[x];
        for (int i = 0; i < x; i++)
        {
            Xp_auto_jet[i].a = prior_state_estimate[i];
            Xp_auto_jet[i].v[i] = 1;
        }
        ceres::Jet<double, x> Yp_auto_jet[y];
        func(Xp_auto_jet, Yp_auto_jet);
        for (int i = 0; i < y; i++)
        {
            prior_state_estimate_measure[i] = Yp_auto_jet[i].a;
            measurement_matrix.block(i, 0, 1, x) = Yp_auto_jet[i].v.transpose();
        }
        measurement_cov_maxtrix = (measurement_matrix * error_cov_post * measurement_matrix.transpose() + measurement_noise_cov).inverse();
        residual = measure_vec - prior_state_estimate_measure;
        if (fabs(residual[1]) > 1.5 * M_PI) // yaw轴残差矫正
        {
            if (residual[1] > 0)
            {
                residual[1] -= 2 * M_PI;
            }
            else
            {
                residual[1] += 2 * M_PI;
            }
        }
        return prior_state_estimate;
    }

    Eigen::Matrix<T, x, 1> update() //根据实际值计算,并更新
    {
        kalman_gain = error_cov_post * measurement_matrix.transpose() * measurement_cov_maxtrix;
        posteriori_state_estimate = prior_state_estimate + kalman_gain * residual;
        error_cov_post = (Eigen::Matrix<T, x, x>::Identity() - kalman_gain * measurement_matrix) * error_cov_post;
        return posteriori_state_estimate;
    }

    double ChiSquaredTest()
    {
        Eigen::Vector2d residual_xy = Eigen::Vector2d(residual[0], residual[1]);
        Eigen::Matrix2d measurement_cov_maxtrix_xy;
        measurement_cov_maxtrix_xy << measurement_cov_maxtrix(0, 0), measurement_cov_maxtrix(0, 1),
            measurement_cov_maxtrix(1, 0), measurement_cov_maxtrix(1, 1);

        return residual_xy.transpose() * measurement_cov_maxtrix_xy * residual_xy;
    }
};
#endif