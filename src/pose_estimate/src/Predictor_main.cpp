#include "Predictor_main.h"
#include "math.h"

namespace ly
{

#define GRAVITY 9.79338
    Predictor::Predictor()
    {
        // 车辆跟随器
        // vehicle_tracker = new VehicleTracking();
        // 扩展卡尔曼滤波
        ekf_filter = new NormalEKF();
        
        if (GlobalParam::SOCKET)
        {
            // 创建了socket连接
            // 端口号3000
            udpsender = new UDPSender("192.168.1.13", 3000);
        }
        // 初始上一帧的位置
        last_pose_vec = {0, 0, 0};
    }

    Predictor::~Predictor()
    {
        // delete vehicle_tracker;
        delete ekf_filter;
    }

    // 两帧识别之间的差值
    Angle_t Predictor::Predict(const Point3d& armor1, const Point3d& armor2, bool is_get_second_armor, int detect_mode, SerialPortData SerialPortData_, float &frame_delta_t)
    {
        static float t_raw = 0.0f;
        static float t_predict = 0.0f;

        // 先根据当前云台pitch和目标距离解算大致击打时间
        // 角度转弧度
        float pitch_gimbal = SerialPortData_.pitch / 100.0f * 3.14159f / 180.0f;
        float yaw_gimbal = SerialPortData_.yaw / 100.0f * 3.14159f / 180.0f;

        // 设置子弹的速度
        ShootSpeed = 15.5f;

        // 获取指向的pitch、yaw以及距离，发射时间
        // DLOG(WARNING) << "                      "  <<  SerialPortData_.pitch;
        // DLOG(WARNING) << "                      "  <<  SerialPortData_.yaw;

        // 没有补偿
        // 实际没有使用 pitch
        shootAngleTime_now = ballistic_equation(pitch_gimbal, armor1);

        // 子弹飞行时间, ms 转换成 s
        float shootTime = shootAngleTime_now.time / 1000;
        t_raw += frame_delta_t;                 //  单位为s   
        double shoot_delay = 110;               //  单位 ms 补偿, 击打时间的补偿  
        
        // 差距主要就在shoottime和shootdelay
        t_predict = t_raw + shootTime + (shoot_delay / 1000.0f);           // 之后写成一个配置, 这个补偿
        DLOG(INFO) << "                         delay: " << shootTime + (shoot_delay / 1000.0f) << "s";
        Eigen::Vector3d predict_point;
        Eigen::Vector3d predict_speed;
        Eigen::Vector3d filte_point;

        Point3d tracked_armor_pose; // 跟踪的装甲板

        // 检测到第二个装甲板
        if (is_get_second_armor)
        {
            // norm()，计算欧拉距离，L2范数
            // 也就是根据距离判断，判断跟踪的装甲板
            Point3d last_armor(last_pose_vec.x,last_pose_vec.y,last_pose_vec.z);
            tracked_armor_pose = (norm(armor1 - last_armor)< norm(armor2 - last_armor)) ? armor1 : armor2;
        }
        else
        {
            tracked_armor_pose = armor1;
        }

        // 跟踪，没什么问题
        Eigen::Vector3d tracked_armor(tracked_armor_pose.x, tracked_armor_pose.y, tracked_armor_pose.z);
        
        // DLOG(INFO) << "t_raw: " << t_raw;
        // DLOG(INFO) << "t_predict: " << t_predict;

        // 滤波得到的三维坐标
        filte_point = ekf_filter->runKalman(tracked_armor, frame_delta_t); // 滤波得到的点 上一个时刻预测该时刻
        
        // DLOG(INFO) << armor1;
        // DLOG(WARNING) << "                                 " << filte_point[0] << ", " << filte_point[1] <<", " << filte_point[2] ;

        // 记录上一个装甲板滤波后的位置
        // last_pose_vec 其中滤波得到的结果，上一次识别可靠度比较高的部分
        last_pose_vec = cv::Point3d(filte_point[0], filte_point[1], filte_point[2]);

        // 预测的位置           射击时间以及延时, 预测那个时刻的位置
        // 预测落点 单位s
        predict_point = ekf_filter->predict(shootTime + (shoot_delay / 1000.0f));                         // 预测下一个位置

        // predict_speed = ekf_filter->getSpeed(); // xyz的speed

        // 获取pitch、yaw、distance
        // Eigen::Vector3d pyd_pre = ekf_filter->getPYD();

        // 预测点的坐标
        Point3d cv_predict_point(predict_point.x(), predict_point.y(), predict_point.z());

        DLOG(WARNING) << "                                 predict: " << predict_point[0] << ", " << predict_point[1] <<", " << predict_point[2] ;

        // 瞄准预测点的位置, 根据预测点进行估计
        shootAngleTime_pre = ballistic_equation(pitch_gimbal, cv_predict_point);

        // DLOG(INFO) << shootAngleTime_pre.distance;

        // // 整车预测解算 // LOST_BUMP状态，进行无量测更新 // CONTINOUS_GET_TARGET状态，正常更新

        // predict_hit_point = vehicle_tracker->predictVehicleState(armor_pose, armor_pose_sec, is_get_second_armor, detect_mode, shootTime, frame_delta_t, yaw_gimbal);
        // shootAngleTime_pre = ballistic_equation(pitch_gimbal, predict_hit_point);

        // 装甲板距离
        // double distance = norm(tracked_armor_pose);
        
        // pitch yaw distance
        // Eigen::Vector3d pyd(pitch_gimbal, yaw_gimbal, distance);

        // 当前位置
        data.x_now = tracked_armor_pose.x;      // 该时刻的测量值, 实际用时是该时刻的滤波值
        data.y_now = tracked_armor_pose.y;      
        data.z_now = tracked_armor_pose.z;

        // data.x_now = tracked_armor_pose.x;      // 该时刻的测量值, 实际用时是该时刻的滤波值
        // data.y_now = tracked_armor_pose.y;      
        // data.z_now = tracked_armor_pose.z;

        // 预测的位置
        data.x_pre = predict_point[0];       // 下一时刻预测值
        data.y_pre = predict_point[1];
        data.z_pre = predict_point[2];
        // data.x_pre = filte_point[0];         // 该时刻的最优估计
        // data.y_pre = filte_point[1];
        // data.z_pre = filte_point[2];         
        // data.x_pre = last_pose_vec.x;           // 上一时刻的最优值估计
        // data.y_pre = last_pose_vec.y;
        // data.z_pre = last_pose_vec.z;

        // 原始的raw
        data.time_raw = t_raw;      // 原始raw时间
        data.time_pre = t_predict;  // 预测predict时间
        if(GlobalParam::SOCKET)
            udpsender->send(data);

        // 返回 预测的 位置
        return shootAngleTime_pre;
    }

    void Predictor::resetPredictor()
    {
        // vehicle_tracker->resetTracker();
    }

    // 根据物理方程来计算设定pitch和yaw
    // gim_pitch pitch角度 armor_Position装甲板坐标
    Angle_t Predictor::ballistic_equation(float gim_pitch, Point3d armor)
    {
        Angle_t shootAngleTime_;

        // 先计算yaw轴角度, 弧度制
        if(armor.y == 0){
            armor.y += 0.000001;
        }
        shootAngleTime_.yaw = atan(-armor.x / armor.y);

        // 距离
        shootAngleTime_.distance = sqrt(armor.x * armor.x + armor.y * armor.y);

        // armor 的位置进行了一定的旋转
        // Eigen::Vector3d armor_new_position = Eigen::Vector3d(0, shootAngleTime_.distance, armor_Position[2]);

        // 计算pitch轴的初始角度
        shootAngleTime_.pitch = atan(armor.z/shootAngleTime_.distance);

        // DLOG(WARNING) << armor; 
        // DLOG(WARNING) << shootAngleTime_.pitch * 100 * 180 / M_PI;
        // DLOG(WARNING) << shootAngleTime_.yaw * 100 * 180 / M_PI;


        // 考虑空气阻力， 计算角度
        double theta = shootAngleTime_.pitch;
        double delta_z;

        // R = 42.50 mm, m = 41 g
        // 首先计算空气阻力系数 K
        double k1 = 0.47 * 1.169 * (2 * 3.14159f * 0.02125 * 0.02125) / 2 / 0.041;
        // 使用迭代法求解炮弹的发射角度
        // v是炮弹的发射速度，英雄默认 15 即可
        // 根据炮弹的初速度、发射角度、空气阻力系数，计算炮弹的飞行轨迹
        // 灯钩
        for (int i = 0; i < 100; i++)
        {
            // 计算炮弹的飞行时间
            double t = (pow(2.718281828, k1 * shootAngleTime_.distance) - 1) / (k1 * ShootSpeed * cos(theta));

            delta_z = armor.z - ShootSpeed * sin(theta) * t / cos(theta) + 4.9 * t * t / cos(theta) / cos(theta);
            // DLOG(INFO) << "delta_y: " << delta_y;

            // 不断更新theta，直到小于某一个阈值
            if (fabs(delta_z) < 0.000001)
                break;

            // 更新角度
            theta -= delta_z / (-(ShootSpeed * t) / pow(cos(theta), 2) + 9.8 * t * t / (ShootSpeed * ShootSpeed) * sin(theta) / pow(cos(theta), 3));
        }

        // 发射角度时间的数据
        shootAngleTime_.pitch = theta; // 调整弹道后的数据

        // 子弹飞行时间 单位 ms
        shootAngleTime_.time = abs(shootAngleTime_.distance / (ShootSpeed * cos(shootAngleTime_.pitch)) * 1000);

        // 将pitch转换成 角度值
        shootAngleTime_.pitch = (shootAngleTime_.pitch) / 3.1415926 * 180.0;

        // 将pitch转换成角度值
        shootAngleTime_.yaw = shootAngleTime_.yaw / 3.1415926 * 180.0f;

        return shootAngleTime_;
    }
}