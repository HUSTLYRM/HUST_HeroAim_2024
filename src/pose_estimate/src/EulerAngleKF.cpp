
#include<EulerAngleKF.h>
namespace ly
{
    EulerAngleKF::EulerAngleKF()
    {
        is_kalman_init = false;
        posteriori_Q4 = Eigen::Vector4d::Zero();
        posteriori_d_Q4 = Eigen::Vector4d::Zero();
        process_noice = Eigen::Matrix4d::Identity();
        process_noise_matrix = Eigen::Matrix<double, 8, 4>::Zero();
        error_cov_post = Eigen::Matrix<double, 8, 8>::Identity();
        Eigen::Vector2d maritx_10;
        //maritx_10 << 1, 0;
        //for (int i = 0; i < 4; i++)
        //{
        //    measurement_matrix.block<1,2>(i,2*i) = maritx_10;
        //}

        // 测量矩阵
        measurement_matrix<<1,0,0,0,0,0,0,0,
                            0,0,1,0,0,0,0,0,
                            0,0,0,0,1,0,0,0,
                            0,0,0,0,0,0,1,0;
        // 过程噪声协方差
        process_noise_cov = Eigen::Matrix<double, 8, 8>::Identity() * 0.001; //表明对预测过程的相信程度
        // 测量噪声协方差
        measurement_noise_cov = Eigen::Matrix<double, 4, 4>::Identity()*100;     //证明对测量过程的相信程度
        posteriori_state_estimate = Eigen::Matrix<double, 8, 1>::Zero();     //后验估计由外部进行初始化
        measurement_cov_maxtrix = Eigen::Matrix<double, 4, 4>::Zero();
        dont_believe_measure=false;
    }

    Eigen::Matrix<double, 8, 1> EulerAngleKF::runKalman(const Eigen::Quaterniond &new_q4, const double &delta_t) //量测有效更新
    {
        if (!is_kalman_init)
        {
            //set signal values
            is_kalman_init = true;

            //reset kalman
            rebootKalman(new_q4);

            //return values
            return posteriori_state_estimate;
        }
        else
        {
            //set update time
            setUpdateTime(delta_t);
            
            //update transition matrix
            setTransitionMatrix();

            // DLOG(WARNING) << transition_matrix << std::endl;

            return correct(new_q4);
        }
    }

    void EulerAngleKF::rebootKalman(const Eigen::Quaterniond &new_q4)
    {
        setMeasurementNoise();
            //后验估计状态向量
            posteriori_state_estimate[0] = new_q4.w();
            posteriori_state_estimate[2] = new_q4.x();
            posteriori_state_estimate[4] = new_q4.y();
            posteriori_state_estimate[6] = new_q4.z();
            
        for (int i = 0; i < 4; i++)
        {
            posteriori_state_estimate[i * 2 + 1] = 0;
            posteriori_Q4[i] = posteriori_state_estimate[i * 2];//后验Q4
            posteriori_d_Q4[i] = posteriori_state_estimate[i * 2 + 1];//后验d_Q4
        }
        //初始化状态转移协方差以及状态转移矩阵
        error_cov_post = Eigen::Matrix<double, 8, 8>::Identity();
        transition_matrix = Eigen::Matrix<double, 8, 8>::Identity();
    }

    Eigen::Matrix<double, 8, 1> EulerAngleKF::correct(const Eigen::Quaterniond &new_q4)
    {
        // 设置系统噪声
        setProcessNoise();
        //setMeasurementNoise();
        if (update_time > 1) //大于0.2s没有观测到数据，选择重启卡尔曼滤波
        {
            LOG(WARNING) << "RESET KALMAN DUE TO TIME";
            rebootKalman(new_q4);
            return posteriori_state_estimate;
        }

        // 进行预测
        predict(new_q4);

        // DLOG(WARNING) <<" 先验状态估计 "<<predict(new_q4) <<std::endl;
        update();

        // DLOG(WARNING)<< "posteriori_state " << posteriori_state_estimate;

        for (int i = 0; i < 4; i++)
        {
            //update armor status and return
            posteriori_Q4[i] = posteriori_state_estimate[i * 2];
            posteriori_d_Q4[i] = posteriori_state_estimate[i * 2 + 1];
        }
        return posteriori_state_estimate;
    }

    void EulerAngleKF::setProcessNoise()
    {
        Eigen::Matrix<double, 2, 1> process_noice_vec;
        process_noice_vec << 0.5 * update_time * update_time, update_time;
        for (int i = 0; i < 4; i++)
        {
            process_noise_matrix.block<2, 1>(2 * i, i) = process_noice_vec;
        }

        process_noice.diagonal() << FilterParams::process_noise_q4_w,
            FilterParams::process_noise_q4_x,
            FilterParams::process_noise_q4_y,
            FilterParams::process_noise_q4_z; //3个轴的过程噪声
        process_noise_cov = process_noise_matrix * process_noice * process_noise_matrix.transpose();
    }
    void EulerAngleKF::setMeasurementNoise()
    {
        //pitch,yaw,distance的噪声
        double measurement_noise_q4_w = 0.00001;
        double measurement_noise_q4_x = 0.00001;
        double measurement_noise_q4_y = 0.00001;
        double measurement_noise_q4_z = 0.00001;

        measurement_noise_cov.diagonal() << measurement_noise_q4_w,
            measurement_noise_q4_x,
            measurement_noise_q4_y,
            measurement_noise_q4_z; 
    }
    void EulerAngleKF::setMeasurementNoise(double w_noise,double x_noise,double y_noise,double z_noise)
    {
        //DLOG(ERROR) << "setMeasurementNoise:  "<< yaw_noise<<"  "<<pitch_noise<<"  "<<roll_noise<<"  "<<std::endl;
        measurement_noise_cov.diagonal() << w_noise,
            x_noise,
            y_noise,
            z_noise; 
    }

    Eigen::Matrix<double, 8, 1> EulerAngleKF::predict(const Eigen::Quaterniond &new_q4)
    {
        // 先验状态估计 = A * 最优状态估计
        // [8, 1] = [8, 8] * [8, 1] 
        prior_state_estimate = transition_matrix * posteriori_state_estimate;
        // 先验误差协方差 P = A * P * AT + R
        // [8, 8] = [8, 8] + [8, 8]
        error_cov_post = transition_matrix * error_cov_post * transition_matrix.transpose() + process_noise_cov;
        
        //measurement_cov_maxtrix = (measurement_matrix * error_cov_post * measurement_matrix.transpose() + measurement_noise_cov).inverse();
        // 先验状态 测量估计 = 先验状态估计中的 旋转四元数部分
        for (int i = 0; i < 4; i++)
        {
            prior_state_estimate_measure[i] = prior_state_estimate[2*i];
        }
        
        // 卡尔曼增益 的 分母
        // [4, 4] = [4,8]*[8,8]*[8,4] + [4,4] 
        measurement_cov_maxtrix = (measurement_matrix * error_cov_post * measurement_matrix.transpose() + measurement_noise_cov).inverse();
        
        // 本次测量值
        Eigen::Vector4d measure_vec;
        measure_vec[0]=new_q4.w();
        measure_vec[1]=new_q4.x();
        measure_vec[2]=new_q4.y();
        measure_vec[3]=new_q4.z();

        // H是 [8,4] 测量矩阵

        // 计算最优估计时的差
        // TODO 这里是不是有问题，少 x 了一个 H
        // 计算后验估计的那一部分 Zk - H * Xk
        // 残差 = 测量值 - 估计值 4
        residual = measure_vec- prior_state_estimate_measure;

        return prior_state_estimate;
    }

    Eigen::Matrix<double, 8, 1> EulerAngleKF::update()
    {
        // K = Pk * H^T * (H * Pk * H^T + R)^-1
        // 计算卡尔曼增益 K [8,4] = [8,8]*[8,4]*[4,4]
        kalman_gain = error_cov_post * measurement_matrix.transpose() * measurement_cov_maxtrix;

        // TODO 卡尔曼滤波这里有问题： 导致计算得到的kalman增益有问题， 完全相信了预测值， 没有相信测量值
        DLOG(WARNING) << "kalman_gain "<< kalman_gain << std::endl;

        // 计算最优状态估计 [8,1] = [8,1] + [8,4]*[4,1]
        // 也就是说这里 完全相信了预测值
        posteriori_state_estimate = prior_state_estimate + kalman_gain * residual;
        
        // 更新 后验误差协方差 P = (I-K*H)* P^
        error_cov_post = (Eigen::Matrix<double, 8,8>::Identity() - kalman_gain * measurement_matrix) * error_cov_post;
        
        // DLOG(WARNING) << "posteriori_state_omega" <<posteriori_state_estimate<< std::endl;
        return posteriori_state_estimate;
    }

    void EulerAngleKF::setUpdateTime(const double &delta_t)
    {
        if (fabs(delta_t) < 1e-4) //防止时间差为0
        {
            update_time = 8.0/1000.0f ;
        }
        else
        {
            update_time = delta_t/1000.0f ;
        }
    }

    void EulerAngleKF::setTransitionMatrix()
    {
        Eigen::Matrix2d transition;
        transition << 1, update_time, 0, 1;
        for (int i = 0; i < 4; i++)
        {
            transition_matrix.block<2, 2>(i * 2, i * 2) = transition;
        }
    }
}