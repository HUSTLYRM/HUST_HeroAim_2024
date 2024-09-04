#include "NormalEKF.h"
#include <fstream>
namespace ly
{   
    // 初始化
    NormalEKF::NormalEKF(/* args */)
    {
        is_kalman_init = false;
        // 状态量 x y z以及xv yv zv
        // 观测量 pitch yaw distance
        kalman_filter = new ExtendedKalman<double, 6, 3>();     // 扩展卡尔曼滤波器 观测值是pitch、yaw、distance
        // 后验状态
        posteriori_pose = Eigen::Vector3d::Zero();
        posteriori_speed = Eigen::Vector3d::Zero();
        // 过程噪声
        process_noice = Eigen::Matrix3d::Identity();
        // 过程噪声矩阵
        process_noise_matrix = Eigen::Matrix<double, 6, 3>::Zero();
    }

    NormalEKF::~NormalEKF()
    {
    }

    // 重启卡尔曼滤波器
    void NormalEKF::rebootKalman(const Eigen::Vector3d &new_armor_pose)
    {
        // 设置状态量
        for (int i = 0; i < 3; i++)
        {
            kalman_filter->posteriori_state_estimate[i * 2] = new_armor_pose[i];
            kalman_filter->posteriori_state_estimate[i * 2 + 1] = 0;

            posteriori_pose[i] = kalman_filter->posteriori_state_estimate[i * 2];
            posteriori_speed[i] = kalman_filter->posteriori_state_estimate[i * 2 + 1];
        }
        // 后验协方差矩阵
        kalman_filter->error_cov_post = Eigen::Matrix<double, 6, 6>::Identity();
        // 重置状态转移矩阵
        resetTransitionMatrix();
    }

    // 重置状态转移矩阵
    void NormalEKF::resetTransitionMatrix()
    {
        // x,x_v,y,y_v,z,z_v
        kalman_filter->transition_matrix = Eigen::Matrix<double, 6, 6>::Identity();
    }

    // 设置更新时间
    void NormalEKF::setUpdateTime(const double &delta_t) // 传进来的数据是s
    {
        if (fabs(delta_t) < 1e-4) // 防止时间差为0
        {
            update_time = 8.0 / 1000.0;
        }
        else
        {
            update_time = delta_t;     
        }
    }

    // 执行卡尔曼滤波器
    Eigen::Vector3d NormalEKF::runKalman(const Eigen::Vector3d &new_armor_pose, const double &delta_t) // 量测有效更新
    {
        if (!is_kalman_init)
        {
            // set signal values
            is_kalman_init = true;

            // reset kalman
            rebootKalman(new_armor_pose);   // 重启卡尔曼

            // return values
            return new_armor_pose;
        }
        else
        {
            // set update time
            setUpdateTime(delta_t);     // 设置更新时间

            // update transition matrix
            setTransitionMatrix();

            return correct(new_armor_pose); // 返回卡尔曼滤波器
        }
    }

    // 设置状态转移矩阵
    void NormalEKF::setTransitionMatrix()
    {
        Eigen::Matrix2d transition;
        transition << 1, update_time, 0, 1;         // x = x + vt
        for (int i = 0; i < 3; i++)
        {
            kalman_filter->transition_matrix.block<2, 2>(i * 2, i * 2) = transition;
        }
    }

    // 获得纠正数据
    Eigen::Vector3d NormalEKF::correct(const Eigen::Vector3d &armor_pose)
    {
        // 设置测量噪声
        setProcessNoise();

        // 设置观测噪声
        setMeasurementNoise(armor_pose); // 设置测量噪声
        
        // 根据位置 得到 pitch yaw distance
        Eigen::Vector3d pyd = measure(armor_pose);

        // std::fstream ss("../pyd.txt", std::ios::app);
        // ss << pyd << std::endl;
        // ss.close();
        // std::cout << "update_time:" << update_time << std::endl;

        if (update_time > 0.3) // 大于0.2s没有观测到数据，选择重启卡尔曼滤波
        {
            DLOG(WARNING) << "RESET KALMAN DUE TO TIME!!!!!!!!!!!!!!!!!!";
            rebootKalman(armor_pose);
            return armor_pose;
        }

        // 卡尔曼滤波进行预测
        kalman_filter->predict(xyz_to_pyd, pyd);        // 使用pitch、yaw、distance进行预测

        // 检验
        detect_param = kalman_filter->ChiSquaredTest(); // 进行卡方检验

        // DLOG(WARNING) << detect_param;
        /********************反小陀螺状态检测开始*******************/
        // if (detect_param > VERIFY_THRESH) // 检验失败
        // {
        //     DLOG(WARNING) << "RESET KALMAN DUE TO VERIFY!!!!!!!!!!!!!!!!!";
        //     rebootKalman(armor_pose);
        //     return armor_pose;
        // }

        // 更新
        kalman_filter->update();

        // 得到后验状态估计
        for (int i = 0; i < 3; i++)
        {
            // update armor status and return
            posteriori_pose[i] = kalman_filter->posteriori_state_estimate[i * 2];
            posteriori_speed[i] = kalman_filter->posteriori_state_estimate[i * 2 + 1];
        }
        return posteriori_pose;
    }

    // 测量
    Eigen::Vector3d NormalEKF::measure(const Eigen::Vector3d &armor_pose)
    {   
        // x y z
        pyd[2] = armor_pose.norm(); // 距离
        pyd[0] = ceres::atan2(armor_pose[2], sqrt(armor_pose[0] * armor_pose[0] + armor_pose[1] * armor_pose[1])); // pitch
        pyd[1] = ceres::atan2(-armor_pose[0], armor_pose[1]);   //  (-x) / y                                         // yaw
        return pyd;
    }

    // 根据测量值设置测量噪声矩阵
    void NormalEKF::setMeasurementNoise(const Eigen::Vector3d &armor_pose)
    {   
        double ratio = 1;

        // pitch,yaw,distance的噪声
        double measurement_noise_pose_pitch = 0.0001 * ratio;
        double measurement_noise_pose_yaw = 0.0001 * ratio;

        double distance = armor_pose.norm();
        double measurement_noise_pose_distance;

        // 根据距离设置不同的噪声
        if (distance < 1.5) // 统计方法计算，分段线性
        {
            measurement_noise_pose_distance = pow(distance * 0.01, 2) ;
        }
        else if (distance < 4.5)
        {
            measurement_noise_pose_distance = pow(0.015 + 0.058 * (distance - 1.5), 2) * ratio;
        }
        else
        {
            measurement_noise_pose_distance = pow(0.189 + 0.03 * (distance - 4.5), 2) * ratio;
        }

        kalman_filter->measurement_noise_cov.diagonal() << measurement_noise_pose_pitch,
            measurement_noise_pose_yaw,
            measurement_noise_pose_distance; // 3个轴的测量噪声，感觉三个轴的噪声需要根据PNP距离来计算
    }
    
    // 设置过程噪声矩阵
    void NormalEKF::setProcessNoise()
    {   

        Eigen::Matrix<double, 2, 1> process_noice_vec;
        
        process_noice_vec << 0.5 * update_time * update_time, update_time;

        for (int i = 0; i < 3; i++)
        {
            process_noise_matrix.block<2, 1>(2 * i, i) = process_noice_vec;
        }

        // 只涉及了过程噪声
        process_noice.diagonal() << FilterParams::process_noise_pose_x,          // 过程噪声
            FilterParams::process_noise_pose_y,
            FilterParams::process_noise_pose_z; // 3个轴的过程噪声
        
        // 通过 过程噪声协方差矩阵
        kalman_filter->process_noise_cov = process_noise_matrix * process_noice * process_noise_matrix.transpose();
    }

    // 设置过程噪声矩阵
    void NormalEKF::setProcessNoise(double x, double y, double z)
    {
        Eigen::Matrix<double, 2, 1> process_noice_vec;
        process_noice_vec << 0.5 * update_time * update_time * 0.1, update_time * 0.1;
        for (int i = 0; i < 3; i++)
        {
            process_noise_matrix.block<2, 1>(2 * i, i) = process_noice_vec;
        }
        process_noice.diagonal() << x, y, z;
        // DLOG(WARNING) << "Q++++++++++++++++++++++++++++++++++++++++++" << process_noise_matrix * process_noice * process_noise_matrix.transpose() << endl;
        kalman_filter->process_noise_cov = process_noise_matrix * process_noice * process_noise_matrix.transpose();
    }

    // 噪声小 收敛快，更相信测量和状态转移方程
    // 噪声大 收敛慢，更加平滑

    // 预测
    Eigen::Vector3d NormalEKF::predict(const double &predict_t)
    {
        return posteriori_pose + posteriori_speed * predict_t;
    }

}