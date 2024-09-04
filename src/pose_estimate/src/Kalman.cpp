// #include "Kalman.h"

// namespace ly
// {

//     /**
//      * Kalman()构造函数
//      * 初始化kalman的 转移矩阵，测量矩阵，
//     */
//     Kalman::Kalman()
//     {   
//         // ^ 表示先验，也可以认为就是预测（x[k-1]表示上一时刻的最佳估计, u[k-1]表示控制向量）
//         // 预测得到先验状态     x[k]^ = A*x[k] + B*u[k-1]   
//         // 得到先验的协方差矩阵  P[k]^ = A*P[k-1]*AT + Q    (系统误差)

//         // 更新卡尔曼增益       K = Pk^ * HT / (H * P[k]^ * HT + R)     (测量误差)
//         // 更新后验状态估计     x[k] = x[k]^ + K (Z[k]-H*X[k]^)         (在更新时需要用到K,这就需要)
//         // 更新误差协方差矩阵   P[k] = (1-K*H)P[k]^

        
//         // 初始胡卡尔曼滤波器 转移矩阵6，测量矩阵6，控制参数
//         KF_ = cv::KalmanFilter(6, 6, 0, CV_32F);
        
//         // 定义测量矩阵 
//         measurement_ = cv::Mat::zeros(6, 1, CV_32F);                    // 包括了6个量 x, y, z, vx, vy, vz
//         // 重置状态转移矩阵 A
//         resetTransitionMatrix();
//         // setIdentity就是设置对角线元素为1 Z = x*H + v 也就是这个H, 得到测量值，根据测量值和预测值求出下一时刻的预测值 
//         setIdentity(KF_.measurementMatrix);

//         // 系统噪声方差矩阵  矩阵Q
//         setIdentity(KF_.processNoiseCov, cv::Scalar::all(0.0001));
//         // 测量矩阵方差矩阵  矩阵R
//         setIdentity(KF_.measurementNoiseCov, cv::Scalar::all(1));
        
//         // 后验错误估计'协方差'矩阵 P (也就是correct后的错误估计协方差矩阵P)
//         setIdentity(KF_.errorCovPost, cv::Scalar::all(1));
//         // 系统初始状态 x[0]
//         randn(KF_.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
//     }

//     Kalman::~Kalman()
//     {
//     }

//     void Kalman::rebootKalman(const ArmorPose &new_armor_pose)
//     {
//         // 重启卡尔曼滤波
//         for (int i = 0; i < 3; i++)
//         {
//             //reset kalman previous estimate and corrected estimate
//             KF_.statePre.at<float>(i) = new_armor_pose[i];
//             KF_.statePre.at<float>(i + 3) = 0.0f;
//             KF_.statePost.at<float>(i) = new_armor_pose[i];
//             KF_.statePost.at<float>(i + 3) = 0.0f;
//         }
//         // 初始化 后验错误估计协方差矩阵 P
//         setIdentity(KF_.errorCovPost, cv::Scalar::all(1));
//         // 初始化 先验错误协方差矩阵 P
//         setIdentity(KF_.errorCovPre, cv::Scalar::all(1));
//         // 重置状态转移矩阵 A
//         resetTransitionMatrix();
//     }
//     // 重置卡尔曼滤波
//     void Kalman::resetKalman()
//     {
//         is_kalman_init = false;
//         is_second_find = false;
//         is_continuous_find = false;
//     }

//     // 开始卡尔曼滤波
//     ArmorPose Kalman::runKalman(const ArmorPose &new_armor_pose, const float &delta_t)
//     {
//         // 没有初始化
//         if (!is_kalman_init)
//         {
//             //set signal values
//             is_kalman_init = true;
//             is_second_find = true;
//             is_continuous_find = false;

//             //reset kalman
//             rebootKalman(new_armor_pose);

//             // 记录该时刻的状态
//             updateArmorState(new_armor_pose);

//             //return values
//             return new_armor_pose;
//         }
//         // 第二次找到
//         else if (is_second_find)
//         {
//             //set signal values
//             is_second_find = false;
//             is_continuous_find = true;

//             //set time
//             setUpdateTime(delta_t);

//             //calculate speed
//             calculateSpeed(new_armor_pose);

//             //not use the predict,correct directly
//             // 个人认为是因为速度还没有初始化，不适合投入使用，第二次检测到说明速度也初始化完成
//             // 更新测量量
//             updateMeasurement(new_armor_pose);  

//             //return the consequence of corrected armor pose
//             // 更新， 计算卡尔曼增益，利用观测量和预测量（此时是初始化的值，上一时刻没有预测），计算最优状态估计, 然后更新误差协方差矩阵， 并记录该时刻的最优估计
//             return correct();
//         }
//         else
//         {
//             //set update time
//             setUpdateTime(delta_t);

//             //update transition matrix 更新状态转移矩阵
//             setTransitionMatrix(delta_t);

//             //calculate armor speed 更新装甲板的速度
//             calculateSpeed(new_armor_pose);

//             //predict by the previous data 开始预测 x[k]^ = A*x[k-1] + B*u[k-1] 得到先验状态估计, 并记录
//             predict();  // 根据上一时刻的测量矩阵和状态矩阵预测该时刻的状态矩阵

//             // 更新测量值
//             updateMeasurement(new_armor_pose);  
            
//             // 首先根据先验的协方差矩阵计算 K, 然后利用测量和先验计算当前的最优状态估计, 然后更新
//             return correct();   // 更新
//         }
//     }
//     // 预测： 得到装甲板位置的先验估计
//     ArmorPose Kalman::predict()
//     {
//         KF_.predict();
//         // 该时刻的估计值
//         // x[k]^ 先验状态估计
//         for (int i = 0; i < 3; i++)
//         {
//             //update armor status and return
//             this_armor_pre_estimate.pose[i] = KF_.statePre.at<float>(i);
//         }
//         for (int i = 0; i < 3; i++)
//         {
//             //update speed and return
//             this_armor_pre_estimate.speed[i] = KF_.statePre.at<float>(i + 3);
//         }
//         return this_armor_pre_estimate.pose;
//     }
    
//     void Kalman::updateArmorState(const ArmorPose &new_armor_pose)
//     {
//         last_armor_state.pose = new_armor_pose;
//     }

//     // 根据预测和测量更新
//     ArmorPose Kalman::correct()
//     {
//         // 根据测量值更新 得到后验状态, 也就是该时刻最优的值
//         KF_.correct(measurement_);
//         // 记录该时刻的状态
//         for (int i = 0; i < 3; i++)
//         {
//             //update armor status and return
//             last_armor_state.pose[i] = KF_.statePost.at<float>(i);
//         }
//         for (int i = 0; i < 3; i++)
//         {
//             //update speed and return
//             last_armor_state.speed[i] = KF_.statePost.at<float>(i + 3);
//         }
//         return last_armor_state.pose;
//     }
//     // 设置两次观测的时间
//     void Kalman::setUpdateTime(const float &delta_t)
//     {
//         if (fabs(delta_t) < 1e-5)
//         {
//             update_time = 15.0f; //if not given the time,to set the time automatically
//         }
//         else
//         {
//             update_time = delta_t;
//         }
//     }

//     // 更新测量矩阵, Z
//     void Kalman::updateMeasurement(const ArmorPose &new_armor_pose)
//     {
//         measurement_ = (cv::Mat_<float>(6, 1) << new_armor_pose[0], new_armor_pose[1],
//                         new_armor_pose[2], this_armor_speed[0], this_armor_speed[1], this_armor_speed[2]);
//     }

//     // 计算速度
//     void Kalman::calculateSpeed(const ArmorPose &new_armor_pose)
//     {
//         //std::cout<<"new_armor_pose: "<<new_armor_pose<<"last_armor_state.pose: "<<last_armor_state.pose<<std::endl;
//         this_armor_speed = (new_armor_pose - last_armor_state.pose) / (update_time / 1000);
//     }

//     // 重置状态转移矩阵， X[t] = A*X[t-1] + B*u[t-1] 没有控制矩阵， 所以 预测时 X[t] = A*X[t-1] 
//     void Kalman::resetTransitionMatrix()
//     {
//         //x,y,z,x_v,y_v,z_v
//         KF_.transitionMatrix = (cv::Mat_<float>(6, 6) << 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
//                                                          0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
//                                                          0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
//                                                          0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
//                                                          0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
//                                                          0.0, 0.0, 0.0, 0.0, 0.0, 1.0 //A 状态转移矩阵
//         );
//     }

//     // 设置状态转移矩阵
//     void Kalman::setTransitionMatrix(float delta_t) //ms
//     {
//         //x,y,z,x_v,y_v,z_v         ms转成s     x'=x+vt
//         KF_.transitionMatrix = (cv::Mat_<float>(6, 6) << 1.0, 0.0, 0.0, delta_t / 1000, 0.0, 0.0,
//                                                          0.0, 1.0, 0.0, 0.0, delta_t / 1000, 0.0,
//                                                          0.0, 0.0, 1.0, 0.0, 0.0, delta_t / 1000,
//                                                          0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
//                                                          0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
//                                                          0.0, 0.0, 0.0, 0.0, 0.0, 1.0 //A 状态转移矩阵
//         );
//     }
// }
