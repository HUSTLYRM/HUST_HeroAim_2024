// //
// // Created by zhiyu on 2022/04/06.
// //

// #include "Predictor.h"
// namespace ly
// {

//     Predictor::Predictor()
//     {   
        
//         if (GlobalParam::SOCKET)
//         {
//             // 创建了socket连接
//             // 端口号5000, 检测卡尔曼滤波
//             udpsender = new UDPSender("192.168.1.3", 5000);
//         }

//         ekf_filter = new NormalEKF();
//         last_pose_vec = {0, 0, 0};


//         // 状态量：x轴z轴的位置，速度，加速度
//         // 预测量: x, z 的位置  （到时候再转变成为yaw和pitch）
//         position_predictor = KalmanFilter(6, 2, 0);        // 状态量：6，预测量：2，控制量：0
        
//         pos_m = Mat::zeros(2, 1, CV_32F);                  // 观测 Z
//         double t = 0.005;
        
//         // 状态转移矩阵 A   6x6
//         position_predictor.transitionMatrix = (Mat_<float>(6, 6) << 
//             1, 0, t, 0, 0.5*t*t, 0,
//             0, 1, 0, t, 0, 0.5*t*t,
//             0, 0, 1, 0, t, 0,
//             0, 0, 0, 1, 0, t,
//             0, 0, 0, 0, 1, 0,
//             0, 0, 0, 0, 0, 1);                              // 状态转移矩阵 A
        
//         // 测量矩阵 H 2x6  ， 没问题
//         setIdentity(position_predictor.measurementMatrix); // 观测矩阵H     Z = H*X + R
//         // setIdentity(position_predictor.processNoiseCov, cv::Scalar::all(1e-6));        // 过程噪声Q

//         // 调参顺序：x->v->a
//         // 可视化出来调
//         // float x_n = 5e-3, z_n = 2e-3;
//         // float v_x = 2e-3, v_z = 1e-1;
//         // float a_x = 1e-1, a_z = 1e-3;

//         float x_n, z_n, v_x, v_z, a_x, a_z;
//         x_n = 5e-3; z_n = 5e-3;
//         v_x = 1e-3; v_z = 1e-3;
//         a_x = 1e-3; a_z = 5e-4;

//         // 过程噪声协方差矩阵     Q     
//         position_predictor.processNoiseCov = (Mat_<float>(6, 6) << x_n*x_n, 0, 0, 0, 0, 0,
//                                                                     0, z_n*z_n, 0, 0, 0, 0,
//                                                                     0, 0, v_x*v_x, 0, 0, 0,
//                                                                     0, 0, 0, v_z*v_z, 0, 0,
//                                                                     0, 0, 0, 0, a_x*a_x, 0,
//                                                                     0, 0, 0, 0, 0, a_z*a_z);
//         // setIdentity(position_predictor.measurementNoiseCov, cv::Scalar::all(1e-4));   // 测量噪声R,0.1
//         // 最好用实测值
//         // 较小的测量噪声，快速收敛
//         float x_m = 1e-2, z_m = 1e-2;

//         // 测量噪声协方差矩阵     R
//         position_predictor.measurementNoiseCov = (Mat_<float>(2, 2) << x_m*x_m, 0,
//             0, z_m*z_m);

//         // 重置后验误差协方差矩阵 P 
//         reset();

//         ////////////////////////////////////////////////
//         //////// 以下是哨兵预测部分，暂时不用

//         // 哨兵预测
//         // sentinel_predictor = KalmanFilter(6, 2, 0);        // 状态量：6，预测量：2，控制量：0
//         // sentinel_m = Mat::zeros(2, 1, CV_32F);   // 观测矩阵H
        
//         // double t = 0.005;
//         // sentinel_predictor.transitionMatrix = (Mat_<float>(6, 6) << 
//         //     1, 0, t, 0, 0.5*t*t, 0,
//         //     0, 1, 0, t, 0, 0.5*t*t,
//         //     0, 0, 1, 0, t, 0,
//         //     0, 0, 0, 1, 0, t,
//         //     0, 0, 0, 0, 1, 0,
//         //     0, 0, 0, 0, 0, 1);                          // 状态转移矩阵A

//         // setIdentity(sentinel_predictor.measurementMatrix); // 观测矩阵H
//         // setIdentity(sentinel_predictor.processNoiseCov, cv::Scalar::all(1e-6));        // 过程噪声Q
//         // 调参顺序：x->v->a
//         // 可视化出来调
//         // x_n = 5e-3; z_n = 2e-3;
//         // v_x = 5e-3; v_z = 1e-3;
//         // a_x = 1e-5; a_z = 1e-3;
//         // x_n = 5e-3; z_n = 1e-2;
//         // v_x = 6e-3; v_z = 1e-4;
//         // a_x = 1e-1; a_z = 1e-5;
//         // sentinel_predictor.processNoiseCov = (Mat_<float>(6, 6) << x_n*x_n, 0, 0, 0, 0, 0,
//         //                                                             0, z_n*z_n, 0, 0, 0, 0,
//         //                                                             0, 0, v_x*v_x, 0, 0, 0,
//         //                                                             0, 0, 0, v_z*v_z, 0, 0,
//         //                                                             0, 0, 0, 0, a_x*a_x, 0,
//         //                                                             0, 0, 0, 0, 0, a_z*a_z);
//         // setIdentity(sentinel_predictor.measurementNoiseCov, cv::Scalar::all(1e-4));   // 测量噪声R,0.1
//         // 最好用实测值
//         // 较小的测量噪声，快速收敛
//         // x_m = 1e-2; z_m = 1e-2;
//         // sentinel_predictor.measurementNoiseCov = (Mat_<float>(2, 2) << x_m*x_m, 0,
//         //     0, z_m*z_m);
//         // setIdentity(sentinel_predictor.errorCovPost, cv::Scalar::all(1e-5));            // 真实噪声P
//         // sentinel_predictor.statePost = (Mat_<float>(1, 6) << 0.1, 2, 0, 0, 0, 0);

//         // 之前的SOCKET部分的代码
//         // if(GlobalParam::SOCKET){
//         //     // 创建了socket连接
//         //     client_socket = socket(AF_INET, SOCK_STREAM, 0);
//         //     server_addr.sin_family = AF_INET;
//         //     server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");       // IP 127.0.0.1是本机回环测试的ip
//         //     server_addr.sin_port = htons(9999);                         // 端口号是9999
//         //     connect(client_socket,(sockaddr *)&server_addr, sizeof(sockaddr));  // 连接
//         // }
//     }

//     Predictor::~Predictor(){
//         // FIXME 暂时注释
//         // 关闭socket连接
//         // close(client_socket);
//     }

//     // 后验状态
//     void Predictor::reset(){
        
//         // 设置后验状态估计 
//         setIdentity(position_predictor.errorCovPost, cv::Scalar::all(1e-6));            // 真实噪声P
//         position_predictor.statePost = (Mat_<float>(1, 6) << 0.1, 2, 0, 0, 0, 0);

//         //////////////////////////////////////////////
//         //  2023已经不需要哨兵单独的预测了

//         // setIdentity(sentinel_predictor.errorCovPost, cv::Scalar::all(1e-6));            // 真实噪声P
//         // sentinel_predictor.statePost = (Mat_<float>(1, 6) << 0.1, 2, 0, 0, 0, 0);
//     }

//     /**
//      * armor 装甲板的3D坐标（x,y,z） 其中y轴是垂直地面的轴!! （一定注意）
//     */
//     double Predictor::fitPNP(const Point3d& armor, bool usePredictor){
//         // position_cur 代表的是 x 和 z 的坐标
//         double d = sqrt(position_cur.at<float>(0)*position_cur.at<float>(0)+position_cur.at<float>(1)*position_cur.at<float>(1));
//         // 根据装甲板得到
//         double r = sqrt(armor.x*armor.x+armor.y*armor.y);   // x方 + y方
//         // 是否使用预测器
//         double ret = usePredictor ? d : r;
//         // 返回预测得到的距离
//         return ret;
//     }

//     /**
//      * @brief 修正弹道，考虑空气阻力以及炮弹等
//      * 
//      * @param armor 
//      * @param v 
//      * @param useRoll 
//      * @return double 
//      */
//     double Predictor::fitTrajectory(const Point3d& armor, double v, bool useRoll){
//         // double theta = atan(armor.y/distance);
//         // double delta_y;
//         // for(int i=0;i<50;i++){
//         //     delta_y = armor.y - distance*tan(theta) + 4.9 * distance*distance/pow(v*cos(theta), 2);
//         //     // DLOG(INFO) << "delta_y: " << delta_y;
//         //     if(fabs(delta_y) < 0.000001) break;
//         //     theta -= delta_y / (- distance / pow(cos(theta), 2) + 9.8 * distance * distance / (v * v) * sin(theta) / pow(cos(theta), 3));
//         // }
//         // if(useRoll) (theta/M_PI*180+distance*0.2)*100;
//         // return (theta/M_PI*180+distance*0.2)*100;
        
//         // 考虑空气阻力， 计算角度
//         double theta = atan(armor.z/distance);
//         double delta_z;

//         // R = 42.50 mm, m = 41 g
//         // 首先计算空气阻力系数 K 
//         double k1 = 0.47*1.169*(2*M_PI*0.02125*0.02125)/2/0.041;
//         // 使用迭代法求解炮弹的发射角度
//         // v是炮弹的发射速度，英雄默认 15 即可
//         // 根据炮弹的初速度、发射角度、空气阻力系数，计算炮弹的飞行轨迹
//         // 灯钩
//         for(int i=0;i<100;i++){
//             // 计算炮弹的飞行时间
//             double t = (pow(2.718281828, k1*distance)-1)/(k1*v*cos(theta));
            
//             delta_z = armor.z - v*sin(theta)*t/cos(theta) + 4.9 * t*t/cos(theta)/cos(theta);
//             // DLOG(INFO) << "delta_y: " << delta_y;
            
//             // 不断更新theta，直到小于某一个阈值
//             if(fabs(delta_z) < 0.000001) break;
            
//             // 更新角度
//             theta -= delta_z / (- (v*t) / pow(cos(theta), 2) + 9.8 * t * t / (v * v) * sin(theta) / pow(cos(theta), 3));
//         }

//         // DLOG(WARNING) << "armor.z" <<armor.z << std::endl;
//         // DLOG(WARNING) << "distance" << distance <<std::endl;
//         // DLOG(WARNING) << "theta"<< theta << std::endl;
//         // DLOG(WARNING) << "theta" << theta/M_PI*180 << std::endl;
//         // DLOG(WARNING) << "delta_y" << delta_y <<std::endl;
        
//         // 默认不适用roll角度，发回去世界坐标系的角度
//         // cos(-roll) ?
//         if(useRoll) return (theta-atan(armor.z/distance))/M_PI*180*100;

//         return (theta/M_PI*180+1)*100;
//     }

//     // 设置后验状态，初始值
//     void Predictor::setStatePost(float x, float z){
//         position_predictor.statePost.at<float>(0) = x;
//         position_predictor.statePost.at<float>(1) = z;
//     }
    
//     // 获取yaw轴数据
//     double Predictor::getYaw(const Point3d& armor, double delta_t, double v, const SerialPortData& imu_data, bool usePredictor){
        
//         // 预测器
//         auto& predictor = position_predictor;
//         // DLOG(INFO) << "state: " << (StateParam::state == SENTINEL ? "sentinel" : "autoaim");

//         // 根据时间更新 状态转移矩阵 A
//         predictor.transitionMatrix = (Mat_<float>(6, 6) << 
//             1, 0, delta_t, 0, 0.5*delta_t*delta_t, 0,
//             0, 1, 0, delta_t, 0, 0.5*delta_t*delta_t,
//             0, 0, 1, 0, delta_t, 0,
//             0, 0, 0, 1, 0, delta_t,
//             0, 0, 0, 0, 1, 0,
//             0, 0, 0, 0, 0, 1);
        
//         // 哨兵模式
//         // if(StateParam::state == SENTINEL){
//         //     sentinel_m.at<float>(0) = armor.x;
//         //     sentinel_m.at<float>(1) = armor.z;
//         //     predictor.correct(sentinel_m);
//         // } else{
//             // 辅瞄模式
        
//         // 设置观测量
//         pos_m.at<float>(0) = armor.x;
//         pos_m.at<float>(1) = armor.z;

//         // 使用观测量 纠正当前的值
//         predictor.correct(pos_m);
//         // }


//         // position_cur 得到预测值 也就是x_n, z_n, x_v, z_n, x_a, z_a
//         // 预测得到的位置
//         position_cur = predictor.predict(); // 预测得到下一时刻的位置以及下一时刻的速度

//         // for both yaw and pitch
//         // distance = fitPNP(armor, false);
//         // 使用预测值的三维距离
//         distance = fitPNP(armor, false);     // true 表示使用预测得到的距离

//         // 记录 射击时间
//         shoot_t = distance/v;

//         // 预测的v*t的值   这里判断一个速度的max为1, 得到的就是
//         // 因为子弹在空中要飞行一定时间，所以要增加这个offset的偏移量
//         // position_cur是当前时刻的预测值
//         // 测量得到的移动偏移， 预测得到的偏移

//         // 预测下一个位置的偏移
//         double offset_x = fmin(fmax(position_cur.at<float>(2), -1), 1)*shoot_t;
//         double offset_z = fmin(fmax(position_cur.at<float>(3), -1), 1)*shoot_t;

//         // position_cur中已经包括了预测的
//         // tmp是保留两位小数后扩大100倍变成整数，得到的（主要是为了和串口通信，收发数据）
//         // 当前要瞄准的位置
//         double tmp = ((atan((position_cur.at<float>(0)+offset_x) / (position_cur.at<float>(1))+offset_z)/M_PI*180))*100;

//         // 如果不使用预测器, pitch轴的数据, 跟随但是不预测

//         // 对于前哨战获取yaw偏移，不使用， 对于
//         if(!usePredictor) {
//             // 所有坐标系都是y向下、x向右、z向前
//             // 装甲板存放的坐标 x 向左，y向上，z向前
//             tmp = ((atan((-armor.x) / (armor.y))/M_PI*180))*100;
//         }

//         // 一个经验性的内容，-0.3， 得到的就是预测的yaw轴的数据
//         // 这是后续调整的内容
//         // 调弹道使用
//         tmp -= 0.8*100;

//         // sqrt
//         double a = sqrt(pow(position_cur.at<float>(4), 2)+pow(position_cur.at<float>(5), 2));

//         // if(a > 4 && StateParam::state == SENTINEL){
//         //     SerialParam::send_data.shootStatus = 0;
//         // }

//         // DLOG(INFO) << "                                           distance: " << distance << " time: " << shoot_t;
//         // DLOG(INFO) << "                                           pred: x " << position_cur.at<float>(0)+offset_x << " z " << position_cur.at<float>(1)+offset_z;
//         // DLOG(INFO) << "                                           speed: x " << position_cur.at<float>(2) << " z " << position_cur.at<float>(3);
//         // DLOG(INFO) << "                                           accel: x " << position_cur.at<float>(4) << " z " << position_cur.at<float>(5);
//         // DLOG(INFO) << "                                           offset: x " << offset_x << " z " << offset_z;

//         // 比较和当前的yaw轴的数据 做一些处理
//         while(abs(tmp - imu_data.yaw) > 9000){
//             if(tmp - imu_data.yaw > 9000){
//                 tmp -= 18000;
//             } else{
//                 tmp += 18000;
//             }
//         }

//         if(GlobalParam::SOCKET){

//             // 输入装甲板 该时刻的装甲板位置
//             aimKFframe.xn = armor.x;
//             aimKFframe.zn = armor.z;
            
//             // 上一时刻的预测
//             // 上一次预测该时刻的位置
//             aimKFframe.pred_xn = last_x;
//             aimKFframe.pred_zn = last_z;

//             // 预测下一时刻的位置，并且考虑了速度的影响
//             // aimKFframe.pred_xn = position_cur.at<float>(0)+offset_x;
//             // aimKFframe.pred_zn = position_cur.at<float>(1)+offset_z;


//             // 本时刻预测的位置-上一时刻预测的x 
//             // aimKFframe.xv = (position_cur.at<float>(0)+offset_x-last_x)/delta_t;
//             // aimKFframe.zv = (position_cur.at<float>(1)+offset_z-last_z)/delta_t;
            
//             // 本时刻预测的位置-上一时刻预测的 / delta_t = 本时刻实际测得的位置
//             // aimKFframe.xv = (armor.x-last_x)/delta_t;
//             // aimKFframe.zv = (armor.z-last_z)/delta_t;
            
//             // 预测的速度 
//             // aimKFframe.pred_xv = position_cur.at<float>(2)+shoot_t*position_cur.at<float>(4);
//             // aimKFframe.pred_zv = position_cur.at<float>(3)+shoot_t*position_cur.at<float>(5);
            
//             // 上一时刻预测的位置
//             // last_x = position_cur.at<float>(0)+offset_x;    // 上一时刻预测的x
//             // last_z = position_cur.at<float>(1)+offset_z;    // 上一时刻预测的z
//             last_x = position_cur.at<float>(0);    // 上一时刻预测的x
//             last_z = position_cur.at<float>(1);    // 上一时刻预测的z


//             // 上一时刻预测的速度
//             last_xv = position_cur.at<float>(0)+offset_x;    // 上一时刻预测的x
//             last_zv = position_cur.at<float>(1)+offset_z;    // 上一时刻预测的z
            

//             udpsender->send(aimKFframe);
//         }
        

        
//         // FIXME 暂时注释
//         // 发送socket内容 调试yaw轴的信息
//         // if(GlobalParam::SOCKET){
//         //     // x            看一下位置的卡尔曼滤波
//         //     buffer = to_string(position_cur.at<float>(0)+offset_x) + " " + to_string(position_cur.at<float>(1)+offset_z) + " " + 
//         //         to_string(armor.x) + " " + to_string(armor.z) + " ";

//         //     // v
//         //     // 预测速度
//         //     // buffer = to_string(position_cur.at<float>(2)+shoot_t*position_cur.at<float>(4)) + " " + to_string(position_cur.at<float>(3)+shoot_t*position_cur.at<float>(5)) + " " + 
//         //     //     to_string((position_cur.at<float>(0)+offset_x-last_x)/delta_t) + " " + to_string((position_cur.at<float>(1)+offset_z-last_z)/delta_t) + " ";
            
//         //     last_x = position_cur.at<float>(0)+offset_x;
//         //     last_z = position_cur.at<float>(1)+offset_z;

//         //     // a
//         //     // buffer = to_string(position_cur.at<float>(4)) + " " + to_string(position_cur.at<float>(5)) + " " + 
//         //     //     to_string((position_cur.at<float>(2)+shoot_t*position_cur.at<float>(4)-last_x)/0.005) + " " + to_string((position_cur.at<float>(3)+shoot_t*position_cur.at<float>(5)-last_z)/0.005) + " ";
//         //     // last_x = position_cur.at<float>(2)+shoot_t*position_cur.at<float>(4);
//         //     // last_z = position_cur.at<float>(3)+shoot_t*position_cur.at<float>(5);

//         //     // buffer = to_string(a) + " " + to_string(position_cur.at<float>(3)+0.1*position_cur.at<float>(5)) + " " + 
//         //     //     to_string((position_cur.at<float>(0)+offset_x-last_x)/0.005) + " " + to_string((position_cur.at<float>(1)+offset_z-last_z)/0.005) + " ";
            
//         //     strcpy(write_str, buffer.c_str());
//         //     write(client_socket, write_str, sizeof(write_str));
//         // }

//         return tmp;
//     }

//     Mat Predictor::getCurState(){
        
//         return position_cur;
//     }

//     Angle_t Predictor::Predict(const Sophus::SE3 &armor_pose, const Sophus::SE3 &armor_pose_sec, bool is_get_second_armor, int detect_mode, SerialPortData SerialPortData_, float &frame_delta_t)
//     {
//         static float t_raw = 0.0f;
//         static float t_predict = 0.0f;

//         // 先根据当前云台pitch和目标距离解算大致击打时间
//         // 角度转弧度, 发送的数据都是按照100倍的过程发送的
//         float pitch_gimbal = SerialPortData_.pitch / 100.0f * 3.14159f / 180.0f;
//         float yaw_gimbal = SerialPortData_.yaw / 100.0f * 3.14159f / 180.0f;

//         // 设置子弹的速度
//         // setShootSpeed(SerialPortData_.shoot_level, 28.5f);
//         ShootSpeed = 15.0f;   

//         // 获取指向的pitch、yaw以及距离，发射时间
//         shootAngleTime_now = ballistic_equation(pitch_gimbal, armor_pose.translation());

//         // 子弹飞行时间, ms 转换成 s
//         float shootTime = shootAngleTime_now.time;  // 

//         // 装甲板检测之间的时间差 单位ms -> s
//         t_raw += frame_delta_t / 1000.0f;
//         // 检测时间差 + 子弹飞行时间 + shootTime 的时间已经是 s 了
// ////////////////////////////// 打个补丁
//         t_predict = t_raw + shootTime + 150; // ms

//         Eigen::Vector3d predict_point;      // 预测位置
//         Eigen::Vector3d predict_speed;      // 预测速度
//         Eigen::Vector3d filte_point;

//         Sophus::SE3 tracked_armor_pose;     // 跟踪装甲板的位置

//         // 检测到第二个装甲板
//         if (is_get_second_armor)
//         {
//             // norm()，计算欧拉距离，L2范数
//             // 也就是根据距离判断，判断跟踪的装甲板
//             tracked_armor_pose = ((armor_pose.translation() - last_pose_vec).norm() < (armor_pose_sec.translation() - last_pose_vec).norm()) ? armor_pose : armor_pose_sec;
//         }
//         else
//         {
//             tracked_armor_pose = armor_pose;
//         }

//         // EKF球面模型测试                      跟踪的装甲板
//         // 
//         filte_point = ekf_filter->runKalman(tracked_armor_pose.translation(), frame_delta_t);

//         // 记录上一个装甲板滤波后的位置
//         last_pose_vec = filte_point;

//         // 预测的位置           射击时间以及延时, 预测那个时刻的位置
//         predict_point = ekf_filter->predict(shootTime + 160); // 传入的是ms
//         predict_speed = ekf_filter->getSpeed();

//         // 获取pitch、yaw、distance
//         Eigen::Vector3d pyd_pre = ekf_filter->getPYD();

//         // predict的位置
//         shootAngleTime_pre = ballistic_equation(pitch_gimbal, predict_point);
//         // LOG(INFO) << armor_pose.translation();
//         // LOG(INFO) << yaw_gimbal;

//         // // 整车预测解算 // LOST_BUMP状态，进行无量测更新 // CONTINOUS_GET_TARGET状态，正常更新
 
//         // predict_hit_point = vehicle_tracker->predictVehicleState(armor_pose, armor_pose_sec, is_get_second_armor, detect_mode, shootTime, frame_delta_t, yaw_gimbal);
//         // shootAngleTime_pre = ballistic_equation(pitch_gimbal, predict_hit_point);


//         // 装甲板距离
//         // double distance = armor_pose.translation().norm();
//         // pitch yaw distance
//         // Eigen::Vector3d pyd(pitch_gimbal, yaw_gimbal, distance);

//         // 当前位置
//         // data.x_now = tracked_armor_pose.translation()[0];
//         // data.y_now = tracked_armor_pose.translation()[1];
//         // data.z_now = tracked_armor_pose.translation()[2];
//         // 预测的位置
//         // data.x_pre = predict_hit_point[0];
//         // data.y_pre = predict_hit_point[1];
//         // data.z_pre = predict_hit_point[2];
//         // data.time_raw = t_raw;
//         // data.time_pre = t_predict;

//         // 获取整车状态
//         // Eigen::Matrix<double, 9, 1> vehicle_state = vehicle_tracker->getVehicleState();

//         // data2.x_c = vehicle_state[0];
//         // data2.y_c = vehicle_state[1];
//         // data2.z_c = vehicle_state[2];
//         // data2.yaw = -vehicle_state[3];
//         // data2.v_x = vehicle_state[4];
//         // data2.v_y = vehicle_state[5];
//         // data2.v_z = vehicle_state[6];
//         // data2.v_yaw = vehicle_state[7];
//         // data2.r = vehicle_state[8];

//         // lcm_debug->sendInfo(armor_pose.translation(), t_raw, predict_point, t_predict);
//         // lcm_debug->sendInfo(pyd, t_raw, pyd_pre, t_predict);

//         return shootAngleTime_pre;
//     }

//     // void Predictor::resetPredictor()
//     // {
//         // vehicle_tracker->resetTracker();
//     // }

//     // 根据物理方程来计算设定pitch和yaw
//     // gim_pitch pitch角度 armor_Position装甲板坐标
//     Angle_t Predictor::ballistic_equation(float gim_pitch, const Eigen::Vector3d &armor_Position)
//     {
//         Angle_t shootAngleTime_;
//         // 先计算yaw轴角度
//         shootAngleTime_.yaw = -atan2(armor_Position[0], armor_Position[1]);
//         // 距离
//         shootAngleTime_.distance = sqrt(armor_Position[0] * armor_Position[0] + armor_Position[1] * armor_Position[1]);
//         // armor 的位置进行了一定的旋转
//         Eigen::Vector3d armor_new_position = Eigen::Vector3d(0, shootAngleTime_.distance, armor_Position[2]);
//         // 计算pitch轴的初始角度
//         shootAngleTime_.pitch = atan2(armor_new_position[2], armor_new_position[1]);

//         float err = 100;

//         float y_temp = armor_new_position[2];
//         float dy, a, y_actual;
//         // 根据弹道方程计算得到
//         for (int i = 0; i < 100; i++)
//         {
//             a = (float)atan2(y_temp, shootAngleTime_.distance);
//             y_actual = BulletModel(shootAngleTime_.distance, ShootSpeed, a);
//             dy = armor_new_position[2] - y_actual;
//             y_temp = y_temp + dy;
//             if (fabs(dy) < 0.001)
//             {
//                 break;
//             }
//         }
//         // 发射角度时间的数据
//         shootAngleTime_.pitch = (float)atan2(y_temp, shootAngleTime_.distance);

//         // 子弹飞行时间 单位 ms
//         shootAngleTime_.time = abs(shootAngleTime_.distance / (ShootSpeed * cos(shootAngleTime_.pitch)) * 1000);

//         shootAngleTime_.pitch = (shootAngleTime_.pitch) / 3.1415926 * 180.0;

//         shootAngleTime_.yaw = shootAngleTime_.yaw / 3.1415926 * 180.0f;

//         return shootAngleTime_;
//     }

//     float Predictor::BulletModel(float x, float v, float angle)
//     {
//         // 空气阻力系数
//         // float init_k_ = 0.47*1.169*(2*M_PI*0.02125*0.02125)/2/0.041;
//         // float t, y;
//         // t = (float)((exp(init_k_ * x) - 1) / (init_k_ * v * cos(angle))); // k系数等待测量
//         // y = (float)(v * sin(angle) * t - GRAVITY * t * t / 2);
//         // return y;
//         return 0;
//     }
// }
