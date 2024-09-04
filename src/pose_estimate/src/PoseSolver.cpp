//
// Created by zhiyu on 2021/8/20.
//

/**
 * @file PoseSolver.cpp   没有移植BA优化的代码
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-04-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "../include/PoseSolver.h"

using namespace std;
namespace ly {
    // BA 优化的状态量
    Eigen::Matrix<double, 8, 1> angle_state_vector;

    PoseSolver::PoseSolver() {

        predictor = new Predictor();            // 创建一个预测器

        send_shoot_time = std::chrono::steady_clock::now();

        // 设置相机内外参数
        setCameraMatrix(CameraParam::fx, CameraParam::fy, CameraParam::u0, CameraParam::v0);

        // 设置相机畸变
        setDistortionCoefficients(CameraParam::k1, CameraParam::k2, CameraParam::p1, CameraParam::p2, CameraParam::k3);

        // 云台坐标系在世界坐标系下的位姿
        gimbal_to_world = Sophus::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0));

        // 相机坐标系 运动成 云台坐标系 （坐标系的变化意味着坐标点的反变换）
        // double degree=-8;

        // 老车这里是5
        double DEG_TO_ARC = 0.0174532925199433;

        double degree_yaw = CameraParam::yaw;
        double degree_pitch = CameraParam::pitch;
        double degree_roll = CameraParam::roll;

        Eigen::Vector3d euler_angle(degree_pitch * DEG_TO_ARC, degree_roll * DEG_TO_ARC, degree_yaw * DEG_TO_ARC);
        Eigen::Matrix3d rotation_matrix;

        // 固定的旋转角度, 旋转固定的角度
        // 先roll
        rotation_matrix = Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitZ()) * // yaw
                          Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY());

        // 相机坐标系在云台坐标系下的位姿
        camera_to_gimbal = Sophus::SE3(rotation_matrix,
                                       Eigen::Vector3d(CameraParam::camera_trans_x, CameraParam::camera_trans_y,
                                                       CameraParam::camera_trans_z));
//        file_1.open("distance_1m.csv", ios::out);
//        file_2.open("euler_1m.csv", ios::out);
        // 创建预测器
        // predictor = new Predictor();
        // bumper = new Bumper();  // 辅瞄目标缓冲器

        // if (GlobalParam::SOCKET)
        // {
        //     // 创建了socket连接
        //     // 端口号3000
        //     udpsender = new UDPSender("192.168.1.3", 3000);
        // }
    }

    // 设置相机内参
    void PoseSolver::setCameraMatrix(double fx, double fy, double u0, double v0) {
        camera_matrix = cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0));
        camera_matrix.ptr<double>(0)[0] = fx;
        camera_matrix.ptr<double>(0)[2] = u0;
        camera_matrix.ptr<double>(1)[1] = fy;
        camera_matrix.ptr<double>(1)[2] = v0;
        camera_matrix.ptr<double>(2)[2] = 1.0f;
    }

    // 设置畸变系数矩阵
    void PoseSolver::setDistortionCoefficients(double k_1, double k_2, double p_1, double p_2, double k_3) {
        distortion_coefficients = cv::Mat(5, 1, CV_64FC1, cv::Scalar::all(0));
        distortion_coefficients.ptr<double>(0)[0] = k_1;
        distortion_coefficients.ptr<double>(1)[0] = k_2;
        distortion_coefficients.ptr<double>(2)[0] = p_1;
        distortion_coefficients.ptr<double>(3)[0] = p_2;
        distortion_coefficients.ptr<double>(4)[0] = k_3;
    }

    // 更新imu的信息数据
    void PoseSolver::setimu(float pitch, float yaw, float roll) {
        Eigen::Matrix3d rotation_matrix3;

        // 云台坐标系 在 世界坐标系 中的位姿



        // 更换了旋转的顺序
        // 绕z轴旋转roll角, z轴朝前
//        pitch -= 1.2 * M_PI / 180.0f;  // 平衡pitch的重力方向
//        roll = 0;
//        rotation_matrix3 =

        rotation_matrix3 =Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX())*// 绕y轴旋转-yaw弧度
                          Eigen::AngleAxisd(-roll, Eigen::Vector3d::UnitY()); // 绕x轴旋转pitch弧度


        // 创建一个平移向量
        // gimbal相对于世界坐标系只有一个旋转，没有平移
        gimbal_to_world = Sophus::SE3(rotation_matrix3, Eigen::Vector3d(0, 0, 0)); // x,y,z平移均为0
    }

    // 余弦公式
    float PoseSolver::cosineLaw(float a, float b, float c) {
        double value = fmin(fmax((a * a + b * b - c * c) / (2 * a * b), -1), 1);
        // DLOG(INFO) << "a: " << a << " b: " << b << " c: " << c << " acos: " << value;
        return acos(value) / M_PI * 180;
    }

    // 更新时间
    void PoseSolver::update_delta_t(double &delta_time) {
        delta_t = delta_time;
    }

    /**
     * @brief
     *
     * @param armor 指定的要解算位姿的装甲板
     * @param imu_data 当前的imu数据，包括pitch、yaw、roll，都是相对于世界坐标系的数据
     *                 pitch、roll数据是根据实际得到的，yaw轴数据是相当于初始上电时的基准来的
     *                 pitch 当前云台与水平面的夹角
     */
    Sophus::SE3 PoseSolver::solveArmor(ArmorBlob &armor, const SerialPortData &imu_data) {
        // DLOG(WARNING) << "solveArmor" <<std::endl;

        // 装甲板的位姿
        const static Sophus::SE3 armor_pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0));

        // DLOG(WARNING) << armor.corners << std::endl;
        // 先不管平衡步兵

        double width = sqrt((armor.corners[0].x - armor.corners[1].x) * (armor.corners[0].x - armor.corners[1].x) +
                            (armor.corners[0].y - armor.corners[1].y) * (armor.corners[0].y - armor.corners[1].y));;
        double height = sqrt((armor.corners[2].x - armor.corners[1].x) * (armor.corners[2].x - armor.corners[1].x) +
                             (armor.corners[2].y - armor.corners[1].y) * (armor.corners[2].y - armor.corners[1].y));
        vector<Point2f> corners_center;
        // corners_center[i] 是 corners[i] 和 corners[(i+1)%4] 的中点
        for (int i = 0; i < 4; i++) {
            corners_center.push_back((armor.corners[i] + armor.corners[(i + 1) % 4]) / 2);
        }
        double ratio = width / height;
        DLOG(INFO) << "ratio: " << ratio << endl;

        // 对corners取中点，ccornersi与
        DLOG(WARNING)<<"armor_class: "<<armor._class<<endl;


        if (armor._class == 1 || armor._class == GlobalParam::BIGID || (StateParam::state==AUTOAIM && armor._class==6 && 0))         // 1号， 超过三倍的按照大装甲板进行解算
        { // 或者等于基地大装甲
            // large armor
            DLOG(WARNING)<< "large armor" << endl;
            armor.is_big_armor = true;
            solvePnP(points_large_3d, armor.corners, camera_matrix, distortion_coefficients,
                     rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
        } else {
            DLOG(WARNING)<< "small armor" << endl;
            armor.is_big_armor = false;
            // small armor
            solvePnP(points_small_3d, armor.corners, camera_matrix, distortion_coefficients,
                     rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
        }
        // DLOG(WARNING) << "----------" << std::endl;
        // DLOG(WARNING) << rvec << std::endl;

        // 平移向量比较重要，旋转向量没那么重要了 经过变换，x轴朝右，y轴超前，z轴朝上

        // double temp = tvec.ptr<double>(0)[1];            // x = x
        // tvec.ptr<double>(0)[1] = tvec.ptr<double>(0)[2]; // y = z
        // tvec.ptr<double>(0)[2] = -temp;                  // z = -y
        // 使用一个SE3矩阵实现
        Eigen::Matrix3d R;
        R << 1, 0, 0,
                0, 0, 1,
                0, -1, 0;

        Sophus::SE3 transRight(R, Vector3d(0, 0, 0));

        // temp = rvec.ptr<double>(0)[1];                      // x = x
        // rvec.ptr<double>(0)[1] = rvec.ptr<double>(0)[2];    // y = z
        // rvec.ptr<double>(0)[2] = -temp;                     // z = -y

        cv2eigen(tvec, e_T); // 平移向量没有问题

        // 旋转
        Rodrigues(rvec, m_R);      // 旋转顺序没有改变
        cv2eigen(m_R, e_R);

        // yaw = atan2(m_R.at<double>(1, 0), m_R.at<double>(0, 0)) / M_PI * 180;
        yaw = atan2(-m_R.at<double>(2, 0), sqrt(pow(m_R.at<double>(2, 1), 2) + pow(m_R.at<double>(2, 2), 2))) / M_PI *
              180;
//         DLOG(WARNING) << "                                                      yaw: " << yaw << std::endl;

        armor.angle = abs(yaw);

        armor_to_camera = transRight * Sophus::SE3(e_R, e_T);


        // 将 armor_pose 从装甲板坐标系转换成相机坐标系, 不使用旋转向量
        // 得到当前的yaw轴数据      角度转成弧度
        float rcv_yaw = imu_data.yaw / 100.0f * M_PI / 180.0f;

        // 得到当前的pitch轴数据    角度转弧度
        float rcv_pitch = imu_data.pitch / 100.0f * M_PI / 180.0f;

        // roll轴的数据， 增加了roll的数据
        // roll不准，不断获取新的roll
        // float rcv_roll = roll.getMetric() / roll.getSize();

        // DLOG(INFO) << "roll: "<< imu_data.roll / 100.0f << std::endl;

        float rcv_roll = imu_data.roll / 100.0f * M_PI / 180.0f;
        // rcv_roll = 0;


        // 更新当前的imu数据, 更新云台 -> 世界坐标系的转化矩阵
        // 暂时不使用angle, 旋转向量
        setimu(rcv_pitch, rcv_yaw, rcv_roll);
        // e_T 平移向量
        // armor_to_camera = transRight * Sophus::SE3(Eigen::Matrix3d::Identity(), e_T) * armor_pose;
        // 从 相机坐标系坐标 转换成 世界坐标系 坐标
        camera_to_world = gimbal_to_world * camera_to_gimbal;
        // 从 装甲板坐标 转换到 云台坐标系 坐标
        armor_to_gimbal = camera_to_gimbal * armor_to_camera;
        // 从 装甲板坐标 转换到 世界坐标系 坐标
        armor_to_world = camera_to_world * armor_to_camera;

        Eigen::Vector3d z = armor_to_world * Eigen::Vector3d(0, 0, 1);  // 世界坐标系下的z轴
        // 求z在xy平面的投影的相位
        double armor_yaw = atan2(z[1], z[0]) / M_PI * 180;
//        DLOG(WARNING) << "                    armor_yaw:" << armor_yaw << std::endl;

        // DLOG(WARNING) << camera_to_world.translation() << std::endl;



        // DLOG(INFO) << "                              corners: " << armor.corners;

        // Eigen::Matrix3d rotation_matrix = armor_to_gimbal.rotation_matrix();
        // cv::eigen2cv(rotation_matrix, m_R);

        // z 角度偏移
        // float anglez = atan2(m_R.at<double>(1, 0), m_R.at<double>(0, 0)) / M_PI * 180;

        // 角度， 相加坐标系下的yaw轴角度
        // FIXME 对这个公式存疑

        // yaw = atan2(-m_R.at<double>(2, 0), sqrt(pow(m_R.at<double>(2, 0), 2) + pow(m_R.at<double>(2, 2), 2))) / M_PI * 180;
        // 角度
        // https://zhuanlan.zhihu.com/p/85108850 根据这张表来看，应该是这个公式
        // [-90 ~ 90]


        // float anglex = atan2(m_R.at<double>(2, 1), m_R.at<double>(2, 2)) / M_PI * 180;

        // DLOG(ERROR) << "armor angle: " << yaw << std::endl;
        // DLOG(ERROR) << "armor pitch: " << anglex << std::endl;
        // DLOG(ERROR) << "【Recv】 hero roll: " << rcv_roll << std::endl;
        // DLOG(INFO) << "z: " << z << " y: " << yaw << " x: " << x;

        // 装甲板在云台坐标系下的坐标
        const auto &a = armor_to_gimbal;
        Point3d t = {a.translation()[0],
                     a.translation()[1],
                     a.translation()[2]};

        // DLOG(INFO) << t;

        // Point3d trans = {-armor_to_world.translation()[0], // x
        //  -armor_to_world.translation()[1], // y
        //  armor_to_world.translation()[2]}; // z

        // 更换坐标系之后的坐标
        Point3d trans = {armor_to_world.translation()[0],  // x
                         armor_to_world.translation()[1],  // y
                         armor_to_world.translation()[2]}; // z

        // DLOG(ERROR) << t << std::endl;
        // DLOG(ERROR) << " world " << trans << std::endl;

        // 确定一下pitch、yaw、roll和rotation() 坐标的对应关系

        // 装甲板的角度 主要是y轴对应的旋转角度
        // armor.angle = yaw;

        // 世界坐标系中的坐标
        armor.x = trans.x; // 装甲板的坐标定义
        armor.y = trans.y;
        armor.z = trans.z;


        DLOG(WARNING) << "armors pos: " << armor.x << " " << armor.y << " " << armor.z << std::endl;
        file_1<<armor.x<<","<<armor.y<<","<<armor.z<<endl;

        //距离
        DLOG(WARNING)<<"distace: "<<sqrt(armor.x*armor.x+armor.y*armor.y+armor.z*armor.z)<<endl;
        // 世界坐标系中的

        // 使用VOFA+ 进行udp通信调试

        // TODO: 测试上交的yaw解算方法
//         double armorWidth = 0.0275f * 2; // 装甲板宽度
//         double armorLength = 0.0675f * 2; // 装甲板长度
//         double armorPitch = 15.0 / 180 * M_PI;
//         vector<cv::Point2f> measurePoints=armor.corners;
//         // 求解平面的法向量，法向量与xy平面夹角为armorPitch，朝向为angle
//
//        Mat frame = cv::Mat::zeros(1024, 1208, CV_8UC3);
//
//        for(int i=0; i<=0; i+=1) {
//            double angle =( i*-20.0 )/180*M_PI;
//            Eigen::Vector3d normalVector;
////            angle = angle / 180 * 3.14159;
//            normalVector << std::cos(angle) * std::cos(armorPitch),
//                    std::sin(angle) * std::cos(armorPitch),
//                    std::sin(armorPitch);
////            cout<<"normalVector: "<<normalVector<<endl;
//            Eigen::Vector3d lengthVector = Eigen::Vector3d(-normalVector.y(), normalVector.x(), 0) / normalVector.head<2>().norm() * armorLength;
//            Eigen::Vector3d widthVector = normalVector.cross(lengthVector) / normalVector.head<2>().norm() * armorWidth;
////            cout<<"lengthVector: "<<lengthVector<<endl;
////            cout<<"widthVector: "<<widthVector<<endl;
//// 世界坐标系到装甲板坐标系的变换矩阵
//            Eigen::Vector3d transEigen;
//            transEigen<<trans.x, trans.y,  trans.z;
//            auto world_to_armor = armor_to_world.inverse();
//            vector<Eigen::Vector3d> vertices;
////            vertices.push_back(transEigen + lengthVector/2 + widthVector/2);
////            vertices.push_back(transEigen + lengthVector/2 - widthVector/2);
////            vertices.push_back(transEigen - lengthVector/2 - widthVector/2);
////            vertices.push_back(transEigen - lengthVector/2 + widthVector/2);
//            vertices.push_back(world_to_armor*(transEigen + lengthVector/2 + widthVector/2));
//            vertices.push_back(world_to_armor*(transEigen + lengthVector/2 - widthVector/2));
//            vertices.push_back(world_to_armor*(transEigen - lengthVector/2 - widthVector/2));
//            vertices.push_back(world_to_armor*(transEigen - lengthVector/2 + widthVector/2));
//            // 画出来四个顶点
//            // 将vertices转成opencv格式Point3d
//            vector<cv::Point3f> verticesCV;
//            for(int i=0; i<4; i++) {
//                cv::Point3f p;
//                p.x = vertices[i].x();
//                p.y = vertices[i].y();
//                p.z = vertices[i].z();
//                verticesCV.push_back(p);
//            }
//
////            cout<<rvec<<endl<<tvec<<endl<<camera_matrix<<endl<<distortion_coefficients<<endl;
//            vector<cv::Point2f> p2d;
////            cout<<verticesCV<<endl;
//            cv::projectPoints(verticesCV, rvec, tvec, camera_matrix, distortion_coefficients, p2d);
////            cout<<"INNNNNNNNNNNN2222333333NNNNNNNNdddddddNNNN2"<<endl;
////            cout<<"                       :"<<p2d[0]<<" "<<p2d[1]<<" "<<p2d[2]<<" "<<p2d[3]<<endl;
//            for(int i=0; i<4; i++) {
//
//                // 利用pnp重投影将p转成图像坐标
//                cv::circle(frame, p2d[i], 2, cv::Scalar(0, 255, 255), 2);
//                cv::line(frame, p2d[i], p2d[(i+1)%4], cv::Scalar(140, 0, 255), 2);
//                // 按序号在旁边标上数字
//                cv::putText(frame, to_string(i), p2d[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
//            }
//
//            // 画出trans的投影
//            // 世界坐标系下的z轴
//            Eigen::Vector3d zAxis = world_to_armor * ( transEigen+Eigen::Vector3d(0, 0, 0.1));
//            // 转换成opencv格式
//            cv::Point3f zAxisCV = Point3f (zAxis.x(), zAxis.y(), zAxis.z());
//            // 从trans出发画出z轴
//            vector<cv::Point3f> zAxisVectorCV;
//            zAxisVectorCV.push_back(Point3f(0,0,0));
//            zAxisVectorCV.push_back(zAxisCV);
//            vector<cv::Point2f> zAxisVector2d;
//            cv::projectPoints(zAxisVectorCV, rvec, tvec, camera_matrix, distortion_coefficients, zAxisVector2d);
////            cout<<"zAxisVector2d: "<<zAxisVector2d[0]<<" "<<zAxisVector2d[1]<<endl;
//            cv::line(frame, zAxisVector2d[0], zAxisVector2d[1], cv::Scalar(0, 255, 0), 2);
//
//
//            // 世界坐标系下的x轴
//            Eigen::Vector3d xAxis = world_to_armor * (transEigen+Eigen::Vector3d(0.1, 0, 0));
//            // 转换成opencv格式
//            cv::Point3f xAxisCV = Point3f (xAxis.x(), xAxis.y(), xAxis.z());
//
//            // 从trans出发画出x轴
//            vector<cv::Point3f> xAxisVectorCV;
//            xAxisVectorCV.push_back(Point3f(0,0,0));
//            xAxisVectorCV.push_back(xAxisCV);
//            vector<cv::Point2f> xAxisVector2d;
//            cv::projectPoints(xAxisVectorCV, rvec, tvec, camera_matrix, distortion_coefficients, xAxisVector2d);
////            cout<<"xAxisVector2d: "<<xAxisVector2d[0]<<" "<<xAxisVector2d[1]<<endl;
//            cv::line(frame, xAxisVector2d[0], xAxisVector2d[1], cv::Scalar(255, 0, 0), 2);
//
//
//            // 世界坐标系下的y轴
//            Eigen::Vector3d yAxis = world_to_armor * (transEigen+Eigen::Vector3d(0, 0.1, 0));
//            // 转换成opencv格式
//            cv::Point3f yAxisCV = Point3f (yAxis.x(), yAxis.y(), yAxis.z());
//            // 从trans出发画出y轴
//            vector<cv::Point3f> yAxisVectorCV;
//            yAxisVectorCV.push_back(Point3f(0,0,0));
//            yAxisVectorCV.push_back(yAxisCV);
//            vector<cv::Point2f> yAxisVector2d;
//            cv::projectPoints(yAxisVectorCV, rvec, tvec, camera_matrix, distortion_coefficients, yAxisVector2d);
////            cout<<"yAxisVector2d: "<<yAxisVector2d[0]<<" "<<yAxisVector2d[1]<<endl;
//            cv::line(frame, yAxisVector2d[0], yAxisVector2d[1], cv::Scalar(0, 0, 255), 2);
//            //画出装甲板坐标系下的z轴
//            vector<cv::Point3f> zAxisVectorArmorCV;
//            zAxisVectorArmorCV.push_back(Point3f(0,0,0));
//            zAxisVectorArmorCV.push_back(Point3f(0,0,0.1));
//
//            vector<cv::Point2f> yAxisVectorArmor2d;
//            cv::projectPoints(zAxisVectorArmorCV, rvec, tvec, camera_matrix, distortion_coefficients, yAxisVectorArmor2d);
//            cv::line(frame, yAxisVectorArmor2d[0], yAxisVectorArmor2d[1], cv::Scalar(255, 255, 0), 2);
//
//            // 世界坐标系下的lengthVector
//            Eigen::Vector3d lengthVectorArmor = world_to_armor * Eigen::Vector3d(0, 1, 0);
//
//            // 从trans出发画出legthVector
////            vector<cv::Point3f> lengthVectorCV;
////            lengthVectorCV.push_back(Point3f(0,0,0));
////            lengthVectorCV.push_back(Point3f(lengthVectorArmor.x(), lengthVectorArmor.y(), lengthVectorArmor.z()));
////            vector<cv::Point2f> lengthVector2d;
////            cv::projectPoints(lengthVectorCV, rvec, tvec, camera_matrix, distortion_coefficients, lengthVector2d);
////            cout<<"lengthVector2d: "<<lengthVector2d[0]<<" "<<lengthVector2d[1]<<endl;
////            cv::line(frame, lengthVector2d[0], lengthVector2d[1], cv::Scalar(255, 0, 0), 2);
//        }
//        // 画出armor的四个corners
//        for(int i=0; i<4; i++) {
//            cv::circle(frame, armor.corners[i], 2, cv::Scalar(0, 0, 255), 2);
//            cv::line(frame, armor.corners[i], armor.corners[(i+1)%4], cv::Scalar(0, 0, 255), 2);
//            cv::putText(frame, to_string(i), armor.corners[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
//        }
//        for(int i=0; i<4; i++) {
//            cv::circle(frame, corners_center[i], 2, cv::Scalar(0, 0, 255), 2);
//            cv::line(frame, corners_center[i], corners_center[(i+1)%4], cv::Scalar(0, 0, 255), 2);
//            cv::putText(frame, to_string(i), corners_center[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
//        }
//        cv::imshow("frame2", frame);
        return armor_to_world; // 暂时不用
    }

    static int offset_x = 432 + 416 / 2;
    static int offset_y = 408 + 416 / 2;


    GimbalPose PoseSolver::solveGimbalPose(Point3d shootTarget) {
        shootTarget.y = (shootTarget.y == 0 ? 1e-6 : shootTarget.y);
        GimbalPose gimbalPose;
        gimbalPose.yaw = atan(-shootTarget.x / shootTarget.y);
        double distance = sqrt(shootTarget.x * shootTarget.x + shootTarget.y * shootTarget.y); // 此处为平面距离
        gimbalPose.pitch = atan(shootTarget.z / distance);

        double theta = gimbalPose.pitch;
        double delta_z;
        // double k1 = 0.47 * 1.169 * (2 * M_PI * 0.02125 * 0.02125) / 2 / 0.041;
        double k1 = 0.47 * 1.169 * (2 * M_PI * 0.02125 * 0.02125) / 2 / 0.041;
        double center_distance = distance;    // 距离
        double gravity = 9.8;
        double flyTime;
        for (int i = 0; i < 100; i++) {
            // 计算炮弹的飞行时间
            flyTime = (pow(2.718281828, k1 * center_distance) - 1) / (k1 * speed * cos(theta));
            delta_z = shootTarget.z - speed * sin(theta) * flyTime / cos(theta) +
                      0.5 * gravity * flyTime * flyTime / cos(theta) / cos(theta);
            if (fabs(delta_z) < 0.000001)
                break;
            theta -= delta_z / (-(speed * flyTime) / pow(cos(theta), 2) +
                                gravity * flyTime * flyTime / (speed * speed) * sin(theta) / pow(cos(theta), 3));
        }

        gimbalPose.pitch = (theta / M_PI * 180) * 100;
        return gimbalPose;
    }

    /**
     * @brief 辅瞄模式, 定点击打模式, 后面可以考虑一下使用卡尔曼滤波进行调整
     *
     * @param armors 装甲板
     * @param delta_t 时间间隔
     * @param imu_data imu数据
     * @param SerialPort_ 串口信息
     */
    bool PoseSolver::getPoseInCamera(vector<ArmorBlob> &armors, double delta_t, const SerialPortData imu_data,
                                     SerialPort *SerialPort_, int &this_frame_class, int &last_frame_class) {
        DLOG(WARNING) << "  >>>>>>>>>>>>>>>            IN FUMIAO            <<<<<<<<<<<<<<<<" << endl;
        static int last_pitch, last_yaw;

        // DLOG(WARNING) << "armors size: " << armors.size() << std::endl;
        if (armors.size() < 1)
            return false;                   // 没有目标

        for (ArmorBlob &a: armors)
            solveArmor(a, imu_data);

        ArmorBlobs candidates;
        // 进入反陀螺模式 ANTITOP
        // if (StateParam::state == ANTITOP)
        // {
        //     // 和上一帧锁定的装甲板编号一样的装甲板 为 候选装甲板
        //     for (const auto &a : armors)
        //     {
        //         if (a._class == last_armor._class)
        //         {
        //             candidates.push_back(a);
        //         }
        //     }

        //     // 此时没有识别到候选装甲板, 一种情况是确实不存在了，还有一种情况是当时没有识别出来（掉帧了）
        //     if (candidates.size() < 1)
        //     {
        //         lost_cnt++; // 掉帧计数
        //         if (lost_cnt > 50)
        //         { // 增加了掉帧缓冲， 这样不会立马退出反陀螺模式
        //             StateParam::state = AUTOAIM;
        //             lost_cnt = 0;
        //             circle.clear(); // 清空 circle （循环数组，存放装甲板的3D坐标）
        //             // DLOG(INFO) << "exit top mode";
        //         }
        //         // DLOG(INFO) << "lost cnt: " << lost_cnt;
        //         return;
        //     }

        //     // 连续识别，清零掉帧计数
        //     lost_cnt = 0;

        //     // 将候选框加入到circle里   相当于一个循环数组
        //     for (const auto &a : candidates)
        //     {
        //         circle.update({a.x, a.y, a.z});
        //     }

        //     // 按照距离上一帧装甲板的距离进行排序
        //     sort(candidates.begin(), candidates.end(), [&](const ArmorBlob &a, const ArmorBlob &b)
        //          { return calcDiff(a, last_armor) < calcDiff(b, last_armor); });

        //     // 找到最接近上一帧装甲板的候选装甲板
        //     armor = candidates.at(0);

        //     // center 这个装甲板的中心坐标
        //     center.x = armor.x;
        //     center.y = armor.y;
        //     center.z = armor.z;

        //     if (circle.getSize() == circle.size())
        //     {
        //         // 获取中心(取了连续100帧相同类别的均值作为中心坐标)
        //         // 计算中心
        //         center = circle.getMetric() / circle.size();
        //         // DLOG(INFO) << "x: " << center.x << " z: " << center.z << " y: " << center.y;

        //         // 遍历候选装甲板 更新选中的装甲板
        //         for (const auto &a : candidates)
        //         {
        //             if (abs(last_armor.angle - a.angle) < 15 && calcDiff(armor, last_armor) <= 0.2)
        //             {
        //                 armor = a;
        //             }
        //         }
        //         // DLOG(INFO) << "last armor - armor: " << abs(last_armor.angle - armor.angle) << " diff: " << calcDiff(armor, last_armor);

        //         // 如果选中的装甲板不满足信息, 就更新退出时间
        //         // TODO: 大于号是不是要改成小于号
        //         if (calcDiff(armor, last_armor) > 0.2)
        //         {
        //             top_exit = std::chrono::steady_clock::now();
        //         }

        //         exit_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - top_exit).count() / 1000.0;
        //         // DLOG(INFO) << "duration " << exit_duration << " top cnt: " << top_cnt;

        //         // 持续时间>2，那么就调整为辅瞄模式AUTOAIM
        //         if (exit_duration > 2)
        //         {
        //             StateParam::state = AUTOAIM;
        //             last_armor = armor;
        //             top_cnt = 0;
        //             return;
        //         }

        //         // 180° 默认小陀螺的转速为0.5
        //         double w = 0.5 * 360;

        //         // 获得与车的水平距离
        //         double distance = sqrt(center.x * center.x + center.z * center.z);

        //         // 估计子弹飞行时间（水平距离）
        //         double time = distance / speed;

        //         // 不 击打
        //         SerialParam::send_data.shootStatus = 0;

        //         // 候选框
        //         for (const auto &a : candidates)
        //         {
        //             // DLOG(INFO) << "center diff: " << abs(a.x - center.x);
        //             // 找到了满足距离中心的条件（反陀螺）
        //             if (abs(a.x - center.x) < 0.02)
        //             {
        //                 shoot_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - top_exit).count() / 1000.0;
        //                 shoot_begin = std::chrono::steady_clock::now();
        //                 // DLOG(INFO) << "        shoot duration: " << shoot_duration;

        //                 thread([this, a, time, w, SerialPort_]() { // 发送数据线程
        //                     int sleep_time = (90 / w - time) * 1000;
        //                     // 休眠一会
        //                     std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        //                     SerialParam::send_data.shootStatus = 1;
        //                     // 开始击打
        //                     SerialPort_->writeData(&SerialParam::send_data);
        //                 })
        //                     .detach();
        //                 break;
        //             }
        //         }
        //     }

        //     // DLOG(INFO) << "                                           x: " << center.x << " y: " << center.y << " z: " << center.z;

        //     // 解算数据
        //     // yaw轴根据卡尔曼的x,y得到, 这里设置的false，不使用预测器，center可以暂时认定是车辆的中心
        //     // 不使用预测器
        //     SerialParam::send_data.yaw = predictor->getYaw(center, delta_t, speed, imu_data, false);

        //     // pitch轴根据解算弹道得到， 直接根据中心和速度估算，不适用roll
        //     SerialParam::send_data.pitch = predictor->fitTrajectory(center, speed);

        //     // DLOG(INFO) << "                                           angle: " << yaw;
        //     // DLOG(INFO) << "                                           send yaw: " << SerialParam::send_data.yaw << " send pitch: " << SerialParam::send_data.pitch;
        //     // DLOG(INFO) << "                                           recv yaw: " << SerialParam::recv_data.yaw << "  recv pitch: " << SerialParam::recv_data.pitch;
        //     // DLOG(INFO) << "                                           x: " << -armor.x << " y: " << -armor.y << " z: " << armor.z;

        //     // 当前绑定的装甲板是中心
        //     last_armor = armor;
        // }

        // ///////////////////////////////////////////
        // //////////////// 以下是辅瞄模式

        // else
        // { // AUTOAIM 辅瞄模式

        right_clicked = imu_data.right_clicked; // 接收到发送的右击数据，右击重置中心装甲板
        if (last_right_clicked == 0 && right_clicked == 1)
            first = true;
        if (first) { // 按下右键时瞄准中心装甲
            first = false;
            // 按照距离中心的距离来进行排序，找到最靠近中心的装甲板
            if (StateParam::state == AUTOAIM_WITH_ROI) {
                // 远距离
                sort(armors.begin(), armors.end(), [](const ArmorBlob &a, const ArmorBlob &b) -> bool {
                    const Rect &r1 = a.rect;
                    const Rect &r2 = b.rect;
                    return abs(r1.x + r1.y + r1.height / 2 + r1.width / 2 - offset_x - offset_y) <
                           abs(r2.x + r2.height / 2 - offset_x + r2.y + r2.width / 2 - 1280 / 2 - offset_y);
                });
            } else {
                // 近距离辅瞄
                sort(armors.begin(), armors.end(), [](const ArmorBlob &a, const ArmorBlob &b) -> bool {
                    const Rect &r1 = a.rect;
                    const Rect &r2 = b.rect;
                    return abs(r1.x + r1.y + r1.height / 2 + r1.width / 2 - 1024 / 2 - 1280 / 2) <
                           abs(r2.x + r2.height / 2 - 1024 / 2 + r2.y + r2.width / 2 - 1280 / 2);
                });
            }
            // 找出距离中心最近的装甲板
            armor = armors.at(0);
            // 最中心的装甲板优先级最高, 操作手瞄准的是最高优先级的装甲板 top_pri 就是操作手开启辅瞄时指定的跟踪目标（最靠近中心的部分）
            top_pri = armor._class;     // 设定优先级
        }

        // 中心最近的装甲板
        // DLOG(WARNING) << "class: " << top_pri << std::endl;
        // 选择装甲板(优先选‘操作手右击离中心最近的’，其次‘选上一次选中的装甲板’，然后‘选择英雄’，最后选择‘步兵’    优先操作手的选项
        int target = chooseArmor(armors);
        DLOG(WARNING)<<"Choose target: "<<target<<endl;
        // 选中的目标
        SerialParam::send_data.num = target; // 当前辅瞄选定的目标

        // 筛选出 优先级最高的 类别
        for (const auto &a: armors) {
            if (a._class == target)
                candidates.push_back(a);
        }

        // 没有符合的装甲板， 可能是掉帧, 减少错误
        if (candidates.size() < 1)
            return false;

        // DLOG(INFO) << "target: " << target << " size: " << candidates.size();
        if (candidates.size() > 1) {
            sort(candidates.begin(), candidates.end(), [&](const ArmorBlob &a, const ArmorBlob &b) {
                return calcDiff(a, last_armor) < calcDiff(b, last_armor);
            });
            // DLOG(INFO) << "diff: " << calcDiff(candidates.at(0), last_armor) << " " << calcDiff(candidates.at(1), last_armor);
        }

        // 选择和上一帧识别的装甲板距离最近的那唯一一个装甲板
        armor = candidates.at(0);

        // candidates 都是装甲板和上一帧锁定的装甲板类别一致的
        // 都更新 last_armor
        if (target == last_armor._class) {
            // 按照和上一帧出现的装甲板距离进行降序排序（相当于一个追踪的效果）

            // 掉帧缓冲
            if (lost_cnt < 20 && calcDiff(armor, last_armor) > 0.3) {   // 如果距离太远就认为进入掉帧, 掉帧之后仍选取上一次的位置作为当前位置
                armor = last_armor;
                lost_cnt++;
            } else {
                lost_cnt = 0;
                armor = candidates.at(0);                               // 更新最近的
            }
            // DLOG(INFO) << "lost cnt: " << lost_cnt;

            bool has_two_armor = false;
            Point3d armor1 = cv::Point3d(candidates[0].x, candidates[0].y, candidates[0].z);
            Point3d armor2 = cv::Point3d(0, 0, 0);
            if (candidates.size() >= 2) {
                has_two_armor = true;
                armor2 = cv::Point3d(candidates[1].x, candidates[1].y, candidates[1].z);
            }

            float delta_time = delta_t; // 类型转换
            Angle_t tmp_angle;
            if (false) {                    // 使用预测
                DLOG(INFO) << "time:    " << delta_time;        // 单位s
                tmp_angle = predictor->Predict(armor1, armor2, has_two_armor, 0, imu_data, delta_time);
                // pitch 已经计算过，增加补偿即可，yaw要重新算

                double tmp_pitch = 0;
                // 得到的pitch和yaw都是角度
                // 5m左右
                DLOG(INFO) << "                                                 distance: " << tmp_angle.distance;
                if (tmp_angle.distance < 4)                                    // 小于4m
                    tmp_pitch = (tmp_angle.pitch + 1) * 100 + 120;
                else if (tmp_angle.distance < 5.5)                              // 小于5.5m
                    tmp_pitch = (tmp_angle.pitch + 1) * 100 + 140;
                else                                                           // 大于5.5m, 6m 以上
                    tmp_pitch = (tmp_angle.pitch + 1) * 100 + 150;

                SerialParam::send_data.pitch = tmp_pitch;

                double tmp_yaw = tmp_angle.yaw * 100;
                tmp_yaw += -0.0 * 100;       // TODO: 进行补偿
                while (abs(tmp_yaw - imu_data.yaw) > 9000) {
                    if (tmp_yaw - imu_data.yaw >= 9000)
                        tmp_yaw -= 18000;
                    else
                        tmp_yaw += 18000;
                }
                SerialParam::send_data.yaw = tmp_yaw;

                SerialParam::send_data.shootStatus = 1;

                DLOG(INFO) << "                                        send yaw: " << SerialParam::send_data.yaw
                           << "  send pitch: " << SerialParam::send_data.pitch;
                DLOG(INFO) << "                                        recv yaw: " << SerialParam::recv_data.yaw
                           << "  recv pitch: " << SerialParam::recv_data.pitch;

                last_armor = armor; // 上一次识别到的armor
                return true;    // 返回跟踪
            }

            // 没有预测，只有跟随
            armor1.y = (armor1.y == 0 ? 1e-6 : armor1.y);
            tmp_angle.yaw = atan(-armor1.x / armor1.y);
            tmp_angle.distance = sqrt(armor1.x * armor1.x + armor1.y * armor1.y + armor1.z * armor1.z);
            tmp_angle.pitch = atan(armor1.z / tmp_angle.distance);

            double theta = tmp_angle.pitch;
            double delta_z;
            // double k1 = 0.47 * 1.169 * (2 * M_PI * 0.02125 * 0.02125) / 2 / 0.041;
            double k1_c = 0.47;
//            if(candidates[0].is_big_armor) {
//                k1_c = 0.70;
//            }
            double k1 = k1_c * 1.169 * (2 * M_PI * 0.02125 * 0.02125) / 2 / 0.041;
            double center_distance = tmp_angle.distance;    // 距离
            if(candidates[0].is_big_armor) center_distance = center_distance / 1.08;
            double gravity = 9.8;
            for (int i = 0; i < 100; i++) {
                // 计算炮弹的飞行时间
                double t = (pow(2.718281828, k1 * center_distance) - 1) / (k1 * speed * cos(theta));
                delta_z = armor1.z - speed * sin(theta) * t / cos(theta) +
                          0.5 * gravity * t * t / cos(theta) / cos(theta);
                if (fabs(delta_z) < 0.000001)
                    break;
                theta -= delta_z / (-(speed * t) / pow(cos(theta), 2) +
                                    gravity * t * t / (speed * speed) * sin(theta) / pow(cos(theta), 3));
            }

            double tmp_pitch = (theta / M_PI * 180) * 100;
//            if(candidates[0].is_big_armor) {
//                tmp_pitch -= 70;
//            }
            // 5m左右
            // if (tmp_angle.distance < 4)                                    // 小于4m
            //     tmp_pitch = tmp_pitch + 100;
            // else if(tmp_angle.distance < 5.5)                              // 小于5.5m
            //     tmp_pitch = tmp_pitch + 140;
            // else                                                           // 大于5.5m, 6m 以上
            //     tmp_pitch = tmp_pitch + 150;

            // >>>> test:
            //          5.1m左右 pitch 偏置 140
            //          3.1m 左右 pitch 偏置 100

            // double tmp_pitch = (theta / M_PI * 180 + 1) * 100 + 150;  // 主要调整的pitch偏置


            // DLOG(INFO) << armor1.z;
            if (GlobalParam::SOCKET) {
//            outpostPoseDataFrame.x = armor.x;
//            outpostPoseDataFrame.y = armor.y;
//            outpostPoseDataFrame.z = armor.z;
                outpostPoseDataFrame.x = imu_data.yaw / 100.;
                outpostPoseDataFrame.y =  tmp_angle.distance ;
                outpostPoseDataFrame.z = SerialParam::recv_data.yaw / 100.;

                outpostPoseDataFrame.pitch = armor.angle;
                outpostPoseDataFrame.yaw = imu_data.yaw;
                outpostPoseDataFrame.roll = imu_data.roll;
//             DLOG(WARNING)<<"Send"<<armor.x<<" "<<armor.y<<" "<<armor.z<<" "<<endl;
                udpsender->send(outpostPoseDataFrame);
            }
            DLOG(INFO) << "distance : " << tmp_angle.distance;
            // 跟随功能

            SerialParam::send_data.shootStatus = 1; // 实际击打

            SerialParam::send_data.pitch = tmp_pitch;

            double tmp_yaw = tmp_angle.yaw / M_PI * 180 * 100;
            tmp_yaw += 0 * 100;       // TODO: 进行补偿
            while (abs(tmp_yaw - imu_data.yaw) > 9000) {
                if (tmp_yaw - imu_data.yaw >= 9000)
                    tmp_yaw -= 18000;
                else
                    tmp_yaw += 18000;
            }
            SerialParam::send_data.yaw = tmp_yaw;

            // DLOG(INFO) << "                                                     " << tmp_angle.yaw * 100;
            // DLOG(INFO) << "                                                     " << tmp_angle.pitch * 100;
            double filter_ratio_yaw = 0.3;
            double filter_ratio_pitch = 0.7;
            DLOG(INFO) << "                                        filter yaw: " << filter_ratio_yaw
                       << "  filter pitch: " << filter_ratio_pitch;

            bool useFilter = !first;
            if(abs(SerialParam::send_data.yaw - last_yaw )> 300 || abs(SerialParam::send_data.pitch - last_pitch) > 100) {
                useFilter = false;
            }
            if(useFilter) {
                SerialParam::send_data.yaw = last_yaw * filter_ratio_yaw + SerialParam::send_data.yaw * (1 - filter_ratio_yaw);
                SerialParam::send_data.pitch = last_pitch * filter_ratio_pitch + SerialParam::send_data.pitch * (1 - filter_ratio_pitch);
            }
            last_yaw = SerialParam::send_data.yaw;
            last_pitch = SerialParam::send_data.pitch;
            DLOG(INFO) << "                                        send yaw: " << SerialParam::send_data.yaw
                       << "  send pitch: " << SerialParam::send_data.pitch;
//            file_2<<SerialParam::send_data.yaw<<","<<SerialParam::send_data.pitch<<endl;
            DLOG(INFO) << "                                        recv yaw: " << SerialParam::recv_data.yaw
                       << "  recv pitch: " << SerialParam::recv_data.pitch;
            last_armor = armor; // 上一次识别到的armor
            return true;
        } else {
            // 掉帧缓冲
            if (lost_cnt < 20)
                lost_cnt++;
            else {
                lost_cnt = 0;
                predictor->resetPredictor();
            }

            last_armor = armor; // 上一次识别到的armor
            first = false;
            return false;
        }

        // 如果距离上一次的装甲板距离过大，认为是反陀螺
        //         // FIXME 这个调小了
        //         // 然后把反陀螺计数多一点

        //         if (calcDiff(armor, last_armor) > 0.2 && fabs(armor.angle - last_armor.angle) > 10) // 同一个目标连续识别到装甲板的距离差别 > 0.2 并且角度超过了10度， 如果只是平移，角度不会超过十度
        //         {
        //             top_cnt++;                                 // 反陀螺计数
        //             predictor->setStatePost(armor.x, armor.z); // 设置后验状态, 测量数据
        //             DLOG(WARNING) << "setStatePost" << std::endl;
        //             top_begin = std::chrono::steady_clock::now();
        //         }

        //         DLOG(INFO) << "last armor - armor: angle: " << fabs(last_armor.angle - armor.angle) << " diff: " << calcDiff(armor, last_armor);

        //         // 1.5s内检测到5次反陀螺，认为进入了反陀螺模式
        //         double duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - top_begin).count() / 1000.0;

        //         if (duration > 1.5)
        //         {
        //             top_cnt = 0;
        //         }
        //         // DLOG(INFO) << "duration " << duration << " top cnt: " << top_cnt;

        //         // 连续检测到5次满足反陀螺条件，转反陀螺模式
        //         if (top_cnt > 5)
        //         { // 每次选择的装甲板类别一样，但是 连续5次识别都不满足一些约束条件，就认为其进入了反陀螺，然后设置状态为反陀螺

        //             // TODO: 后续在这里开启反陀螺模式，现在只开启普通的跟随模式

        //             // StateParam::state = ANTITOP; // 反陀螺模式
        //             top_exit = std::chrono::steady_clock::now();
        //             top_cnt = 0;
        //             // DLOG(INFO) << "top mode";
        //             SerialParam::send_data.shootStatus = 0; // 暂停射击（给反陀螺一定的收敛时间）
        //         }

        //         // 识别到和上一帧一样的装甲板就进行击打
        //         else
        //         {
        //             SerialParam::send_data.shootStatus = 1; // 开始射击（AUTOAIM射击）
        //         }
        //     }
        //     else
        //     {
        //         // 上一帧锁定的目标和当前确定的目标不一致
        //         // DLOG(INFO) << "lost cnt: " << lost_cnt;
        //         // DLOG(INFO) << "no same class.";

        //         // 重置预测器
        //         predictor->reset();

        //         // 掉帧缓冲
        //         if (lost_cnt < 30 && calcDiff(armor, last_armor) > 0.1)
        //         {
        //             armor = last_armor;
        //             lost_cnt++;
        //         }
        //         else
        //         {
        //             lost_cnt = 0;

        //             // 根据候选框的绝对距离排序
        //             sort(candidates.begin(), candidates.end(), [](const ArmorBlob &a, const ArmorBlob &b) -> bool
        //                  { return a.x * a.x + a.z * a.z < b.x * b.x + b.z * b.z; });

        //             // 取出候选框中最近的那个目标, 取出距离自己最近的那个装甲板
        //             armor = candidates.at(0);
        //         }
        //     }

        // 装甲板的3D坐标
        // cur = Point3d(armor.x, armor.y, armor.z);

        // if()

        // 普通辅瞄使用的预测器
        // Angle_t tmp = predictor->Predict();
        // SerialParam::send_data.yaw = predictor->getYaw(cur, delta_t, speed, imu_data);      // 默认yaw轴是考虑预测的，使用false表示不考虑预测

        //     // 启用弹道拟合 求出pitch轴数据
        //     SerialParam::send_data.pitch = predictor->fitTrajectory(cur, speed); // 默认roll轴是不考虑

        //     // 使用VOFA+ 进行udp通信调试
        //     // if (GlobalParam::SOCKET)
        //     // {
        //     //     outpostPoseDataFrame.x = armor.x;
        //     //     outpostPoseDataFrame.y = armor.y;
        //     //     outpostPoseDataFrame.z = armor.z;
        //     //     outpostPoseDataFrame.pitch = SerialParam::send_data.pitch;
        //     //     outpostPoseDataFrame.yaw = SerialParam::send_data.yaw;
        //     //     // outpostPoseDataFrame.roll = 0;
        //     //     udpsender->send(outpostPoseDataFrame);
        //     // }

        //     DLOG(INFO) << "                                           right clicked: " << right_clicked;
        //     DLOG(INFO) << "                                           angle: " << yaw;
        //     DLOG(INFO) << "                                           send yaw: " << SerialParam::send_data.yaw << " send pitch: " << SerialParam::send_data.pitch;
        //     DLOG(INFO) << "                                           recv yaw: " << SerialParam::recv_data.yaw << "  recv pitch: " << SerialParam::recv_data.pitch;
        //     DLOG(INFO) << "                                           recv yaw: " << SerialParam::recv_data.yaw << "  recv pitch: " << SerialParam::recv_data.pitch;
        //     DLOG(INFO) << "                                           x: " << -armor.x << " y: " << -armor.y << " z: " << armor.z;

        //     // 记录下本次选中的装甲板
        //     last_armor = armor;
        //     last_right_clicked = right_clicked;
        // }
        // DLOG(INFO) << "target class: " << armor._class;
    }

    Point2f PoseSolver::antiTop(vector<ArmorBlob> &armors, double delta_t, const SerialPortData &imu_data,
                                SerialPort *SerialPort_, bool &getCenter) {
        DLOG(WARNING) << "  >>>>>>>>>>>>>>>            IN ANTITOP            <<<<<<<<<<<<<<<<" << endl;
        DLOG(INFO)<<"outpost state: "<<(int)SerialParam::recv_data.outpost_state<<endl;
        DLOG(INFO)<<"user bias: "<<(int)SerialParam::recv_data.user_time_bias <<endl;
        static int frameCount = 0;
        static int lostInShootZone = 0;
        getCenter = false;
        for (ArmorBlob &a: armors)
            solveArmor(a, imu_data);

        ArmorBlobs candidates;
//        right_clicked = imu_data.right_clicked; // 接收到发送的右击数据，右击重置中心装甲板
        right_clicked=SerialParam::recv_data.right_clicked;
//        DLOG(WARNING)<<"last_right_clicked: "<<(int)last_right_clicked<<" "<<"right_clicked: "<<(int)right_clicked<<endl;
        if(right_clicked == 0) {
            getCenter = false;
            last_right_clicked = right_clicked;
            return Point2f (0,0);
        }
        if (last_right_clicked == 0 && right_clicked == 1)
            first = true;
        last_right_clicked = right_clicked;
        if (first) { // 只要第一次按右键就清空
            first = false;
            armors_set.clear();
            DLOG(WARNING) << ">>>>>>>>>>>init ANTITOP<<<<<<<<<<<<<" << endl;
            timeInZone.clear();
            time.clear();
            getCenter = false;
            return Point2f(0,0);
        }
        if (armors.size() < 1) {
            getCenter = false;
            return Point2f(0, 0);
        }
        if (first) { // 按下右键时瞄准中心装甲
            // 按照距离中心的距离来进行排序，找到最靠近中心的装甲板

            // 近距离辅瞄
            sort(armors.begin(), armors.end(), [](const ArmorBlob &a, const ArmorBlob &b) -> bool {
                const Rect &r1 = a.rect;
                const Rect &r2 = b.rect;
                return abs(r1.x + r1.y + r1.height / 2 + r1.width / 2 - 1024 / 2 - 1280 / 2) <
                       abs(r2.x + r2.height / 2 - 1024 / 2 + r2.y + r2.width / 2 - 1280 / 2);
            });
            // 找出距离中心最近的装甲板
            armor = armors.at(0);
            // 最中心的装甲板优先级最高, 操作手瞄准的是最高优先级的装甲板 top_pri 就是操作手开启辅瞄时指定的跟踪目标（最靠近中心的部分）
            top_pri = armor._class;     // 设定优先级
        }
        // 中心最近的装甲板
        // DLOG(WARNING) << "class: " << top_pri << std::endl;
        // 选择装甲板(优先选‘操作手右击离中心最近的’，其次‘选上一次选中的装甲板’，然后‘选择英雄’，最后选择‘步兵’    优先操作手的选项
        int target = chooseArmor(armors);
        // 选中的目标
        SerialParam::send_data.num = target; // 当前辅瞄选定的目标
        DLOG(WARNING)<<"target: "<<target<<endl;
        // 筛选出 优先级最高的 类别
        for (const auto &a: armors) {
            if (a._class == target)
                candidates.push_back(a);
        }

        // 没有符合的装甲板， 可能是掉帧, 减少错
        if (candidates.size() < 1) {
//            getCenter = false;

            return Point2f(0, 0);
        }

        DLOG(INFO)<<"armors set:" <<armors_set.size()<<endl;

        if(armors_set.size()>5) {
            getCenter=true;
        }
        else getCenter= false;
        DLOG(INFO) << "armor size: " << candidates.size();
        if (candidates.size() > 1) {
//            sort(candidates.begin(), candidates.end(), [&](const ArmorBlob &a, const ArmorBlob &b) {
//                return calcDiff(a, last_armor) < calcDiff(b, last_armor);
//            });
            // DLOG(INFO) << "diff: " << calcDiff(candidates.at(0), last_armor) << " " << calcDiff(candidates.at(1), last_armor);
            // 找出y最小的装甲板
            sort(candidates.begin(), candidates.end(), [](const ArmorBlob &a, const ArmorBlob &b) {
                return a.y < b.y;
            });
        }

        // 选择和上一帧识别的装甲板距离最近的那唯一一个装甲板
        armor = candidates.at(0);

        DLOG(WARNING)<<"armor_id: "<< armor._class<<endl;
        // candidates 都是装甲板和上一帧锁定的装甲板类别一致的
        // 都更新 last_armor
        if (target == last_armor._class) {
            // 按照和上一帧出现的装甲板距离进行降序排序（相当于一个追踪的效果）

            // 掉帧缓冲
            if (lost_cnt < 20 && calcDiff(armor, last_armor) > 0.3) {   // 如果距离太远就认为进入掉帧, 掉帧之后仍选取上一次的位置作为当前位置
                armor = last_armor;
                lost_cnt++;
            } else {
                lost_cnt = 0;
                armor = candidates.at(0);                               // 更新最近的
            }
            // DLOG(INFO) << "lost cnt: " << lost_cnt;

            bool has_two_armor = false;
            Point3d armor1 = cv::Point3d(candidates[0].x, candidates[0].y, candidates[0].z);
            Point3d armor2 = cv::Point3d(0, 0, 0);
            if (candidates.size() >= 2) {
                has_two_armor = true;
                armor2 = cv::Point3d(candidates[1].x, candidates[1].y, candidates[1].z);
            }

            float delta_time = delta_t; // 类型转换
            Angle_t tmp_angle;
            // 如果candidates[0]的y比shootcenter高超过5cm，不加入
            static Point3d shootCenter = Point3d(0, 0, 0);
//            if (candidates[0].y - shootCenter.y > 0.08 && norm(shootCenter) != 0 && armors_set.size()>25 && candidates.size() == 1) {
//                getCenter = false;
//                return Point2f(0, 0);
//            }
            DLOG(WARNING)<<"center y diff: "<<candidates[0].y - shootCenter.y<<endl;
            if(has_two_armor) {
                DLOG(WARNING)<<"center y diff of armor 2:"<<candidates[1].y-shootCenter.y<<endl;
            }
            armors_set.push_back(candidates[0]);
            // 获取现在的时间戳
            auto now = std::chrono::steady_clock::now();
            // 获取armors中前20帧近距离的装甲板，取平均值获得中心点，当多于20帧时删除最远的
            frameCount++;
            DLOG(INFO)<<"armors_set size: "<<armors_set.size()<<endl;
            if (armors_set.size() > 31) {
                // 通过迭代器找到最大的元素，删除该元素
                auto max = max_element(armors_set.begin(), armors_set.end());
                // 找到最后一个元素，删除该元素
                armors_set.erase(armors_set.end());
                armors_set.erase(max);
            }
            DLOG(WARNING) << "              OUT ANTITOP" << endl;
            for (auto a: armors_set) {
                shootCenter += Point3d(a.x, a.y, a.z);
            }
            if (armors_set.size() != 0)
                shootCenter = shootCenter / (int) armors_set.size();
            // 显示shootCenter


            /////////////////////弹道公式
            // 没有预测，只有跟随

            shootCenter.y = (shootCenter.y == 0 ? 1e-6 : shootCenter.y);
            tmp_angle.yaw = atan(-shootCenter.x / shootCenter.y);
            tmp_angle.distance = sqrt(shootCenter.x * shootCenter.x + shootCenter.y * shootCenter.y);
            tmp_angle.pitch = atan(shootCenter.z / tmp_angle.distance);

            double theta = tmp_angle.pitch;
            double delta_z;
            // double k1 = 0.47 * 1.169 * (2 * M_PI * 0.02125 * 0.02125) / 2 / 0.041;
            double k1 = 0.47 * 1.169 * (2 * M_PI * 0.02125 * 0.02125) / 2 / 0.041;
            double center_distance = tmp_angle.distance / 0.9;    // 距离
            double gravity = 9.8;
            double flyTime;
            for (int i = 0; i < 100; i++) {
                // 计算炮弹的飞行时间
                flyTime = (pow(2.718281828, k1 * center_distance) - 1) / (k1 * speed * cos(theta));
                delta_z = shootCenter.z - speed * sin(theta) * flyTime / cos(theta) +
                          0.5 * gravity * flyTime * flyTime / cos(theta) / cos(theta);
                if (fabs(delta_z) < 0.000001)
                    break;
                theta -= delta_z / (-(speed * flyTime) / pow(cos(theta), 2) +
                                    gravity * flyTime * flyTime / (speed * speed) * sin(theta) / pow(cos(theta), 3));
            }

            double tmp_pitch = (theta / M_PI * 180) * 100;


            DLOG(WARNING) << "           timeInZone size:" << timeInZone.size() << endl;
            static bool markLast; // 用于记录是否标记了上一个的时间
            if (armors_set.size() >= 30) {
                // 计算candidates与shootCenter的距离
                // 空间距离
//                double distance = sqrt(
//                        pow(candidates[0].x - shootCenter.x, 2) + pow(candidates[0].y - shootCenter.y, 2) +
//                        pow(candidates[0].z - shootCenter.z, 2));

                // 使用重投影到平面内平面距离判定击打
                Point2f shootCenter2D= reproject(shootCenter);
                Point2f armor2D = reproject(Point3d(candidates[0].x, candidates[0].y, candidates[0].z));
//                double distance = sqrt(pow(shootCenter2D.x - armor2D.x, 2) + pow(shootCenter2D.y - armor2D.y, 2));
                // 更侧重x轴方向上的距离，设置一个权重 p

                double p = 0.9;
                if(abs(imu_data.roll)>500) p=0.7;
                double distance = p * fabsf(shootCenter2D.x - armor2D.x) + (1 - p) * fabsf(shootCenter2D.y - armor2D.y);

//                distance = distance * center_distance / 5.4

                DLOG(WARNING) << "shoot distance: " << distance << endl;
                // 只考虑xy平面内的距离
//                double distance = sqrt(
//                        pow(candidates[0].x - shootCenter.x, 2) + pow(candidates[0].y - shootCenter.y, 2)
//                );
                if (lostInShootZone >= 3 && !markLast) {
                    std::chrono::steady_clock::duration sum = std::chrono::steady_clock::duration::zero();
                    for (const auto &timestamp: time) {
                        sum += timestamp.time_since_epoch();
                    }
                    if (time.size()) {
                        std::chrono::steady_clock::duration average = sum / time.size();
                        auto average_ms = std::chrono::duration_cast<std::chrono::milliseconds>(average);
                        timeInZone.push_back(average_ms);
                    }
                    time.clear();
                    markLast = true;
                }
                // 当装甲板的位置与center距离小于0.05m认为进入了击打范围
//                DLOG(WARNING) << "distance: " << distance << endl;
                if (distance < 5.0) {
                    DLOG(WARNING) << "IN SHOOT ZONE!!!!" << endl;

                    markLast = false;
                    lostInShootZone = 0;
                    // 将当前时间加入time
                    time.push_back(now);
                    // 拟合一圈开始，识别出了四个装甲板，开始计算时间差
                    if (timeInZone.size() >= 3) {
                        // 计算timeInZone相邻时间之间差值的平均值
                        std::chrono::steady_clock::duration sum = std::chrono::steady_clock::duration::zero();

//                        for (int i = timeInZone.size() - 3; i < timeInZone.size()-1; i++) { // 第一个时间还在拟合不准确，不用
//                            sum += timeInZone[i + 1] - timeInZone[i];
//                            DLOG(INFO)<<(double)timeInZone[i].count()<<" "<<(double)timeInZone[i+1].count()<<" "<<(double)(timeInZone[i + 1].count() - timeInZone[i].count())<<endl;
//                        }
                        sum = timeInZone[timeInZone.size()-1] - timeInZone[timeInZone.size()-3];
                        std::chrono::steady_clock::duration average = sum / 2; // 只用最近两个周期的时间进行拟合
                        auto average_ms = std::chrono::duration_cast<std::chrono::milliseconds>(average);
                        // 设置延迟击打
                        SerialParam::send_data.shootStatus = 0;
                        // 休眠时间，下两个装甲板转过来的时间提前一个发弹延迟的时间
                        bool shootable = 1;
                        if(average_ms.count()>950) shootable=0; // 可能存在掉帧导致记录了两个装甲板的时间
                        DLOG(WARNING) << "average time: " << average_ms.count() << "; tmp_time: " << tmp_time
                                      << "; flyTime: " << flyTime * 1000 << "; user bias: "<< (int)SerialParam::recv_data.user_time_bias * 25<< endl;
                        int i=0, sleep_time=-1;

                        double timeBias;
                        if(SerialParam::recv_data.outpost_state==0) timeBias=time_bias;
                        else timeBias=time_bias_inverse;

//                        timeBias = time_bias;

                        while (sleep_time <0) {
                            sleep_time = (int) (1.0 * i * average_ms.count()  - 1. * tmp_time - flyTime * 1000 + timeBias + (int)SerialParam::recv_data.user_time_bias * 25);
                            i++;
                        }
                        // 设置打弹
                        if(shootable)
                            thread([this, sleep_time, SerialPort_]() {
                                // 延迟击打
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
                                SerialParam::send_data.shootStatus = 1;
                                SerialPort_->writeData(&SerialParam::send_data);
                                DLOG(INFO) << "                shoot";
                            }).detach();

                    }
                }
                    // 距离大于0.2认为离开了击打范围
                else if (distance > 25.0) {
                    lostInShootZone++;
                }
            }

            // 判断这次的装甲板和上次装甲板的距离，判断是否发生了切换
//            DLOG(WARNING)<<"                      armor diff:"<<calcDiff(armor, last_armor)<<endl;
//            if (calcDiff(armor, last_armor) > 0.1) {
//                //将time取平均作为上一个装甲板的时刻
//                std::chrono::steady_clock::duration sum = std::chrono::steady_clock::duration::zero();
//                for (const auto& timestamp : time) {
//                    sum += timestamp.time_since_epoch();
//                }
//                std::chrono::steady_clock::duration average = sum / time.size();
//                auto average_ms= std::chrono::duration_cast<std::chrono::milliseconds>(average);
//
//                timeInZone.push_back(average_ms);
//
//                // 如果发生了切换，将time清空
////                time.clear();
//            }

            //



            // 5m左右
            // if (tmp_angle.distance < 4)                                    // 小于4m
            //     tmp_pitch = tmp_pitch + 100;
            // else if(tmp_angle.distance < 5.5)                              // 小于5.5m
            //     tmp_pitch = tmp_pitch + 140;
            // else                                                           // 大于5.5m, 6m 以上
            //     tmp_pitch = tmp_pitch + 150;

            // >>>> test:
            //          5.1m左右 pitch 偏置 140
            //          3.1m 左右 pitch 偏置 100

            // double tmp_pitch = (theta / M_PI * 180 + 1) * 100 + 150;  // 主要调整的pitch偏置


            // DLOG(INFO) << armor1.z;
            DLOG(INFO) << "distance : " << tmp_angle.distance;
            // 跟随功能

            SerialParam::send_data.shootStatus = 0; // 实际击打

            SerialParam::send_data.pitch = tmp_pitch;

            double tmp_yaw = tmp_angle.yaw / M_PI * 180 * 100;
            tmp_yaw += 0.8 * 100;       // TODO: 进行补偿
            while (abs(tmp_yaw - imu_data.yaw) > 9000) {
                if (tmp_yaw - imu_data.yaw >= 9000)
                    tmp_yaw -= 18000;
                else
                    tmp_yaw += 18000;
            }
            SerialParam::send_data.yaw = tmp_yaw;

            // DLOG(INFO) << "                                                     " << tmp_angle.yaw * 100;
            // DLOG(INFO) << "                                                     " << tmp_angle.pitch * 100;
            DLOG(INFO) << "                                        send yaw: " << SerialParam::send_data.yaw
                       << "  send pitch: " << SerialParam::send_data.pitch;
            DLOG(INFO) << "                                        recv yaw: " << SerialParam::recv_data.yaw
                       << "  recv pitch: " << SerialParam::recv_data.pitch;

            last_armor = armor; // 上一次识别到的armor
            return reproject(shootCenter);
        } else {
            getCenter = false;
            // 掉帧缓冲
            if (lost_cnt < 20)
                lost_cnt++;
            else {
                lost_cnt = 0;
                predictor->resetPredictor();
            }

            last_armor = armor; // 上一次识别到的armor
            return Point2f(0, 0);
        }
        return Point2f(0, 0);
    }

    Point2f PoseSolver::greenShoot(vector<Rect2f> green, double delta_t, const SerialPortData &imu_data,
                                   SerialPort *SerialPort_,  bool &getCenter) {
        DLOG(WARNING) << "  >>>>>>>>>>>>>>>            IN GREENSHOOT            <<<<<<<<<<<<<<<<" << endl;
        DLOG(INFO)<<"Green size: "<<green.size()<<endl;
        const int startColdFrame=30;//only use for the first time
        const int ongoingColdFrame=0;//mean no cold frame
        const Point2f target(770.154, 713.692);//the target you want to adjust to
        // 向左 x 加
        // 向上 y 加

        static int coldingFrame=0;
        int mode=0;
        if(coldingFrame<=0)
        {
            if(!green.empty())
            {
                DLOG(INFO)<<green[0]<<endl;
                Point2f greenTarget=(green[0].tl()+green[0].br())/2.0;
                DLOG(INFO) << "greenTarget: " << greenTarget << endl;
                SerialParam::send_data.shootStatus = 1;
                if(adjustToPoint(greenTarget,target,imu_data,mode, true))
                    DLOG(WARNING) << "ADJUST COMPLETED" << endl;
                getCenter=true;
            }

            if((mode<=1)&&(!green.empty()))
                coldingFrame=startColdFrame;
            else
                coldingFrame=ongoingColdFrame;
        }
        else
        {
            coldingFrame--;
        }
        if(green.empty())
            return Point2f(0,0);
        else
            return (green[0].tl()+green[0].br())/2.0;
    }


    /**
     * @brief 调整相机姿态使得目标点与相机上给定一点重合
     * @param greenTarget 目标点
     * @param cameraTarget 相机上给定的点
     * @param imu_data imu数据
     * @param modeNum 模式编号
     * @param quickStart 是否快速开始调整(默认为false)
     * @param restart 是否重新开始调整(默认为false)
     * @return 是否调整完成
     */
    bool PoseSolver::adjustToPoint(Point2f greenTarget, Point2f cameraTarget, const SerialPortData &imu_data,
                                   int& modeNum,bool quickStart,bool restart) {
        //DLOG(WARNING) << "  >>>>>>>>>>>>>>>            IN ADJUSTOPOINT            <<<<<<<<<<<<<<<<" << endl;
        //算法：先计算lengthRatio等参数(小车摆动固定角度yaw=yawAdjust)(mode=0,1)，然后计算调整量*ChangeRatio(mode=2).
        //可以调整的超参数：lengthRatioDefault,yawAdjust,*ChangeRatio,resAccept

        static int mode=0;
        if(quickStart)
            mode=2;
        static Point2f oldGreenTarget;
        const double lengthRatioDefault=2240.36;//this value shouldn't change even if distance changes, but it change during test :(
        static double lengthRatio=lengthRatioDefault;//use default value
        double yawCurrent=0;
        static double yawSlope=0;//this value should always be 0
        double pitchCurrent=0;
        static double pitchAntiSlope=0;//this value should always be 0

        const double PI=3.14159265358979323846;
        const Point2f screenCenter(1440/2.0,1080/2.0);
        const Point2f center=screenCenter;
        const int yawAdjust=1000;
        const int pitchAdjust=1000;//not used yet
        double yawChangeRatio=1.0;//比例系数
        double pitchChangeRatio=1.0;//比例系数
        const double resAccept=5.0;//Δx^2+Δy^2
        const double shootAccept=15.0;
        if(restart)
        {
            DLOG(WARNING) << "RESTART ENABLED" << endl;
            mode=0;
            lengthRatio=0;
            yawSlope=0;
            pitchAntiSlope=0;
        }

        DLOG(INFO) << "lengthRatio: " << lengthRatio << endl;
        DLOG(INFO) << "yawSlope: " << yawSlope << endl;
        DLOG(INFO) << "yawCurrent: " << yawCurrent << endl;
        DLOG(INFO) << "pitchCurrent: " << pitchCurrent << endl;
        modeNum=mode;
        //region 计算参数
        switch(mode)
        {
            //region 先测试yaw方向参数
            case 0:
            {
                oldGreenTarget=greenTarget;
                if(greenTarget.x<screenCenter.x)
                {
                    SerialParam::send_data.yaw=imu_data.yaw+yawAdjust;
                    SerialParam::send_data.pitch=imu_data.pitch;
                }
                else
                {
                    SerialParam::send_data.yaw=imu_data.yaw-yawAdjust;
                    SerialParam::send_data.pitch=imu_data.pitch;
                }
                mode++;
                break;
            }
            case 1:
            {//x=Ratio*tan(yaw)解二次方程得到
                double x1=oldGreenTarget.x-center.x;
                double x2=greenTarget.x-center.x;
                double deltaThetaTan=tan(yawAdjust*PI/180.0/100);
                double temp=(x2-x1)/deltaThetaTan/2;
                lengthRatio=temp+sqrt(temp*temp-x1*x2);//减的情况为另一根，应该是增根
                if(fabs(lengthRatio-lengthRatioDefault)>200)//防止出现错误的Ratio
                {
                    lengthRatio=lengthRatioDefault;
                    mode=0;
                    return false;
                }
                yawCurrent=atan(x2/lengthRatio);
                yawSlope=(greenTarget.y-oldGreenTarget.y)/(greenTarget.x-oldGreenTarget.x);
                //注：理论上算得Ratio后可直接解出pitch
                pitchAntiSlope=-yawSlope;
                pitchCurrent=atan((greenTarget.y-center.y)/lengthRatio);
                //接下来计算调整量，在switch外计算
                mode++;
                DLOG(INFO) << "lengthRatio: " << lengthRatio << endl;
                DLOG(INFO) << "yawSlope: " << yawSlope << endl;
                DLOG(INFO) << "yawCurrent: " << yawCurrent << endl;
                DLOG(INFO) << "pitchCurrent: " << pitchCurrent << endl;
                break;
            }//endregion
/*            //region 测试pitch方向参数
            case 2:
            {
                if(greenTarget.y<screenCenter.y)
                {
                    SerialParam::send_data.pitch=imu_data.pitch+pitchAdjust;
                }
                else
                {
                    SerialParam::send_data.pitch=imu_data.pitch-pitchAdjust;
                }
                mode++;
                break;
            }
            case 3:
            {
                double y1=oldGreenTarget.y-center.y;
                double y2=greenTarget.y-center.y;
                double deltaThetaTan=tan(pitchAdjust);
                double temp=(y2-y1)/deltaThetaTan/2;
                lengthRatio=temp+sqrt(temp*temp-y1*y2);
                pitchCurrent=atan(y2/lengthRatio);
                pitchAntiSlope=(greenTarget.x-oldGreenTarget.x)/(greenTarget.y-oldGreenTarget.y);
                mode++;
                break;
            }//endregion*/

        }//endregion

        //region 进行调整
        double res=(cameraTarget.x-greenTarget.x)*(cameraTarget.x-greenTarget.x)
                   +(cameraTarget.y-greenTarget.y)*(cameraTarget.y-greenTarget.y);
        DLOG(INFO) << "res: " << res << endl;

        if((mode>=2)&&(res<shootAccept)) {
            SerialParam::send_data.shootStatus=true;
            DLOG(INFO)<<"SHOOT ACCEPTABLE: "<<res<<endl;
        }
        else {
            SerialParam::send_data.shootStatus=false;
        }

        if((mode>=2)&&(res>resAccept))
        {
            //正交分解
            double equalTargetX=(cameraTarget.x+yawSlope*cameraTarget.y)/(1+yawSlope*yawSlope);
            double equalTargetY=cameraTarget.y-yawSlope*equalTargetX;
            yawCurrent=atan((greenTarget.x-center.x)/lengthRatio);
            double yawTarget=atan((equalTargetX-center.x)/lengthRatio);
            pitchCurrent=atan((greenTarget.y-center.y)/lengthRatio);
            double pitchTarget=atan((equalTargetY-center.y)/lengthRatio);

            SerialParam::send_data.yaw=imu_data.yaw+(int)((yawTarget-yawCurrent)*180/PI*100)*yawChangeRatio;
            SerialParam::send_data.pitch=imu_data.pitch+(int)((pitchTarget-pitchCurrent)*180/PI*100)*pitchChangeRatio;
            DLOG(INFO) << "yawCurrent: " << yawCurrent << endl;
            DLOG(INFO) << "pitchCurrent: " << pitchCurrent << endl;
            DLOG(INFO) << "yawTarget: " << yawTarget << endl;
            DLOG(INFO) << "pitchTarget: " << pitchTarget << endl;
        }
        else if(res<=resAccept)
        {
            //mode=0;
            DLOG(INFO) << "Adjustment completed" << endl;
            return true;
        }//endregion

        return false;
    }

    static int shoot_count = 0;
    /**
     * @brief 前哨站模式 前哨站的中心是不断更新的，所以尽管说静止还是什么情况，都可以实现， 前哨站最慢 1/3 /(0.2r/s) = 1.5s  1.5s/5ms = 300, 所以说前哨站中心点的瞄准应该在300帧而不是100帧，才会更稳定一点
     *
     * @param armors
     * @param delta_t
     * @param imu_data
     * @param SerialPort_
     */
    Point2f PoseSolver::outpostMode(vector<ArmorBlob> &candidates, double delta_t, const SerialPortData &imu_data,
                                    SerialPort *SerialPort_, bool &getTarget) {
        DLOG(WARNING) << "  >>>>>>>>>>>>>>>            IN OUTPOST            <<<<<<<<<<<<<<<<" << endl;
        Point2f center_pixel = Point2f(0, 0);

        if(candidates.size() < 1)
            return center_pixel;
        for (ArmorBlob &a : candidates)
            solveArmor(a, imu_data);

        right_clicked = imu_data.right_clicked;
        // DLOG(INFO) << right_clicked;
        if (last_right_clicked == 0 && right_clicked == 1)
            first = true;

        // 第一次右击
        if (first)
        {
            outpost.clear();
            outpost_center.clear();
            filtered_pitch.clear();         // 清空pitch
            first = false;
            shoot_count = 0;

        }

        DLOG(ERROR) << right_clicked;
        last_right_clicked = right_clicked;

        // int outpost_arr_maxsize = outpost.getSize();    // 最多
        for (const auto &a : candidates) // 首先估计中心
        {
            if (outpost.getSize() == outpost.size()){               // 当没满的时候重新对准
                int outpost_arr_cursize = outpost.size();           // 当前
                center = outpost.getMeanOfNearestPoints(min(33, outpost_arr_cursize), outpost_arr_cursize>3 ? 3:0);      // 获取最近的一定的装甲板
                // 在靠近中心一定范围内才会取更新中心
                bool flag = abs(imu_data.roll) < 550 ? abs(a.angle) < 10 : true;

                if (sqrt((a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y) + (a.z - center.z) * (a.z - center.z)) < 0.1 && flag ){   // 距离在一定范围内的都加入
                    outpost_center.update({a.x, a.y, a.z});     // 随后就是只选取其中的一部分进行, 不断选择离我最近的一段始终靠近
                    outpost.update({a.x, a.y, a.z});
                }
            }
            else
                outpost.update({a.x, a.y, a.z});
        }

        double w = 0.4 * 360;

        sort(candidates.begin(), candidates.end(), [&](const ArmorBlob &a, const ArmorBlob &b) { return calcDiff(a, last_armor) < calcDiff(b, last_armor); });

        int outpost_arr_cursize = outpost.size();       // 当前

        // 最初瞄向最近的那个位置

        cv::Point3d minimum = outpost.getMeanOfNearestPoints(min(33, outpost_arr_cursize), outpost_arr_cursize>3 ? 3:0);
        // DLOG(WARNING) << outpost_arr_cursize;
        // DLOG(WARNING) << minimum;

        center.x = minimum.x;
        center.y = minimum.y;
        center.z = minimum.z;

        // 当已经经理过一轮之后
        if (outpost.getSize() == outpost.size())
        {
            getTarget = true;
            if(outpost_center.size() < 10)
                center = outpost.getMeanOfNearestPoints(min(33, outpost_arr_cursize), outpost_arr_cursize>3 ? 3:0);         // 获取中心
            else
                center = outpost_center.getMetric() / outpost_center.size();        // 根据部分平均值获取历史数据

            // 距离
            double distance = sqrt(center.x * center.x + center.y * center.y);

            // 计算时间
            // speed按15.5
            double time = distance / (speed*cos((imu_data.pitch/100.0) * M_PI / 180));

            // 默不射击
            SerialParam::send_data.shootStatus = 0;

            // 选择装甲板准备进行 延时击打
            for (const auto &a : candidates){

                double armor_outpost_distance = sqrt((center.x - a.x) * (center.x - a.x) + (center.y - a.y) * (center.y - a.y) + (center.z - a.z) * (center.z - a.z));
                // 保留中心点
                // Mat center_matrix = cv::Mat(3, 1, CV_64FC1, cv::Scalar::all(0));

                // // 获取中心点的世界坐标系的 xyz, 转到正坐标系
                // center_matrix.ptr<double>(0)[0] = center.x;
                // center_matrix.ptr<double>(1)[0] = center.y;
                // center_matrix.ptr<double>(2)[0] = center.z;

                // Eigen::Vector3d e_center;
                // cv::cv2eigen(center_matrix, e_center);
                // Sophus::SE3 center_pose(Eigen::Matrix3d::Identity(), e_center);

                // // 对于rotation 转置 == 求逆
                // Eigen::Matrix3d rotation = camera_to_world.rotation_matrix().inverse();

                // // world_to_camera 世界坐标系在相机坐标系的位置
                // Sophus::SE3 world_to_camera(rotation, Eigen::Vector3d(-camera_to_world.translation()[0],
                //                                                           -camera_to_world.translation()[1],
                //                                                           -camera_to_world.translation()[2]));

                // // 得到相机坐标系中的前哨站中心三维坐标
                // Sophus::SE3 center_in_camera = world_to_camera * center_pose;

                // Mat T_matrix;
                // cv::eigen2cv(center_in_camera.translation(), T_matrix);

                // double temp = T_matrix.ptr<double>(0)[2];                // x = x
                // T_matrix.ptr<double>(0)[2] = T_matrix.ptr<double>(0)[1]; // z = y
                // T_matrix.ptr<double>(0)[1] = -temp;                      // y = -z

                // // 利用相机内参转换
                // center_matrix = camera_matrix * T_matrix;
                // // uv 坐标
                // double u = center_matrix.at<double>(0, 0) / center_matrix.at<double>(2, 0);
                // double v = center_matrix.at<double>(1, 0) / center_matrix.at<double>(2, 0);

                // bool flag = false;

                // // 2d 装甲板中心点
                // Point2f armor_pixel_center = Point2f(a.rect.x + a.rect.width / 2, a.rect.y + a.rect.height / 2);

                // double pixel_distance = sqrt((armor_pixel_center.x - u) * (armor_pixel_center.x - u) + (armor_pixel_center.y - v) * (armor_pixel_center.y - v));

                // // 装甲板长度
                // double width = sqrt((a.corners[0].x - a.corners[1].x) * (a.corners[0].x - a.corners[1].x) + (a.corners[0].y - a.corners[1].y) * (a.corners[0].y - a.corners[1].y));

                // flag = pixel_distance < width * OutpostParam::center_ratio;

                // DLOG(INFO) << armor_outpost_distance;
                // DLOG(INFO) << pixel_distance;

                // flag = true;
                // 暂时不用
                // 增加三维坐标，避免距离太远，进一步减少误打
                DLOG(WARNING) << armor_outpost_distance << std::endl;
                // DLOG(WARNING) << pixel_distance << std::endl;
                // DLOG(WARNING) << width * OutpostParam::center_ratio << std::endl;

                // flag = true;

                // 角度解算不准，就先这样吧

                // 仍然使用三维数据作为推理

                bool flag = abs(imu_data.roll) < 550 ? abs(armor.angle) < 3 : true;

                if ( armor_outpost_distance < 0.065 && (outpost_center.size() > 5) && flag) {    // 二维图像命中，并且距离在一定范围内

                    if( std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - send_shoot_time).count()/1000 > 500 )
                        shoot_count += 1;

                    send_shoot_time = std::chrono::steady_clock::now();

                    // 休眠时间
                    // double tmp_time = 430;
                    double tmp_time = (120 / w - time) * 1000 + OutpostParam::time_bias + (int)imu_data.user_time_bias * 25; // ms 还要考虑到机械这方面控制的时间

                    if(abs(imu_data.roll) > 550)
                        tmp_time += 30;
                    // DLOG(WARNING) << "whirl_time" << 120 / w *1000 << "ms" <<std::endl;
                    // DLOG(WARNING) << "wait_time: " << tmp_time << "ms" <<std::endl;
                    // DLOG(WARNING) << "fly_time: " << time*1000 << "ms" << std::endl;
                    // DLOG(WARNING) << "shoot_time: " << OutpostParam::time_bias << "ms" << std::endl;

                    // 距离4.5m
                    //  whirl: 833.333 ms
                    //  wait: 429 ms    438 ms  437 ms   430 ms   （418 ms打不中）
                    //  fly: 293 ms
                    //  shoot: -110 ms （也就是手动偏置）
                    thread([this, a, tmp_time, w, SerialPort_]()
                           {
                               int sleep_time = tmp_time;
                               std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
                               SerialParam::send_data.shootStatus = 1;
                               // 暂时不打蛋
                               SerialPort_->writeData(&SerialParam::send_data);
                               DLOG(INFO) << "                shoot"; }).detach();
                    break;
                }

            }
        }
        // DLOG(WARNING) << shoot_count << std::endl;
        double tmp_yaw = ((atan((-center.x) / (center.y)) / M_PI * 180)) * 100;

        // 5m 左右偏置  yaw -0.2度
        //             pitch + 1.2或1.1度
        // 3m 左右可以用
        double distance = sqrt(center.x * center.x + center.y * center.y + center.z * center.z);

        // yaw 的偏置没问题

//        tmp_yaw -= 1.6 * 100;       // TODO: 进行补偿
        while (abs(tmp_yaw - imu_data.yaw) > 9000)        {
            if (tmp_yaw - imu_data.yaw >= 9000)
                tmp_yaw -= 18000;
            else
                tmp_yaw += 18000;
        }

        SerialParam::send_data.yaw = tmp_yaw;

        // 二维地图上的距离
        double center_distance = sqrt(center.x * center.x + center.y * center.y);
        // DLOG(WARNING)
        DLOG(INFO) << "                                        center_cord     :" << center << std::endl;
        DLOG(INFO) << "                                        center_distance :" << center_distance << std::endl;
        // DLOG(ERROR) << "                                        center_z        :" << center.z << std::endl;

        // 考虑空气阻力， 计算pitch
        double theta = atan(center.z / center_distance);
        double delta_z;


        // R = 42.50 mm, m = 41 g
        // 首先计算空气阻力系数 K
        double k1 = 0.47 * 1.169 * (2 * M_PI * 0.02125 * 0.02125) / 2 / 0.041;
        // 使用迭代法求解炮弹的发射角度
        // v是炮弹的发射速度，英雄默认 15 即可
        // 根据炮弹的初速度、发射角度、空气阻力系数，计算炮弹的飞行轨迹
        // 灯钩
        for (int i = 0; i < 100; i++){
            // 计算炮弹的飞行时间
            double t = (pow(2.718281828, k1 * center_distance) - 1) / (k1 * speed * cos(theta));
            delta_z = center.z - speed * sin(theta) * t / cos(theta) + 4.9 * t * t / cos(theta) / cos(theta);
            if (fabs(delta_z) < 0.000001)
                break;
            theta -= delta_z / (-(speed * t) / pow(cos(theta), 2) + 9.8 * t * t / (speed * speed) * sin(theta) / pow(cos(theta), 3));
        }

        short tmp_pitch = (theta / M_PI * 180) *100;

//        if(abs(imu_data.roll) > 550)
//            tmp_pitch -= 50;

        // if (abs(center.z) < 0.3)                                    // 小于0.3m 高度
        // tmp_pitch = (theta / M_PI * 180 + 1) * 100 + 100;
        // else                                                        // >= 0.3m高度
        // tmp_pitch = (theta / M_PI * 180 + 1) * 100 + OutpostParam::pitch_bias;  // 主要调整的pitch偏置

//        if (center_distance < 4)                                    // 小于4m
//            tmp_pitch += 110;
//        else if(center_distance < 5.5)                              // 小于5.5m
//            tmp_pitch += 155;
//        else                                                           // 大于5.5m, 6m 以上
//            tmp_pitch += 165;


        filtered_pitch.update(tmp_pitch);
        SerialParam::send_data.pitch = filtered_pitch.getMetric() / filtered_pitch.size();  // 平均值

        // 前哨站模式下进行调试
        // 使用VOFA+ 进行udp通信调试
        // if (GlobalParam::SOCKET)
        // {
        //     centerFrame.x = center.x;
        //     centerFrame.y = center.y;
        //     centerFrame.z = center.z;
        //     centerFrame.distance = sqrt(center.x * center.x + center.y * center.y + center.z * center.z);
        //     centerFrame.pitch = SerialParam::send_data.pitch;
        //     centerFrame.yaw = SerialParam::send_data.yaw;
        //     udpsender->send(centerFrame);
        // }

        // DLOG(INFO) << "                                           angle: " << yaw;
        DLOG(INFO) << "                                           send yaw: " << SerialParam::send_data.yaw << " send pitch: " << SerialParam::send_data.pitch;
        DLOG(INFO) << "                                           recv yaw: " << SerialParam::recv_data.yaw << "  recv pitch: " << SerialParam::recv_data.pitch << "  recv roll: " << imu_data.roll;
        // DLOG(INFO) << "                                           x: " << center.x << " y: " << center.y << " z: " << center.z;
        last_armor = armor;

        // 绘制前哨站中心
        if (GlobalParam::DEBUG_MODE)
        {

            //  重投影 计算前哨战中心点部分

            // 1. 已知： 前哨战中心的三维坐标
            //          相机和世界坐标系的转换关系
            //          相机内参矩阵
            // 2. 未知:  通过逆操作，求出相机坐标系中的前哨战的坐标

            // 世界坐标系  の  前哨站中心点
            Mat center_matrix = cv::Mat(3, 1, CV_64FC1, cv::Scalar::all(0));
            center_matrix.ptr<double>(0)[0] = center.x;
            center_matrix.ptr<double>(1)[0] = center.y;
            center_matrix.ptr<double>(2)[0] = center.z;
            // center_matrix.ptr<double>(3)[0] = 1.0f;

            // DLOG(INFO) << "前哨战中心点的真实坐标: " << center_matrix << std::endl;

            // std::cout << center_matrix << std::endl;

            // 相机中心在世界坐标系中的坐标
            // Mat camera2world = cv::Mat(3, 1, CV_64FC1, cv::Scalar::all(0));
            // camera2world.ptr<double>(0)[0] = camera_to_world.translation()[0];
            // camera2world.ptr<double>(1)[0] = camera_to_world.translation()[1];
            // camera2world.ptr<double>(2)[0] = camera_to_world.translation()[2];
            // camera2world.ptr<double>(3)[0] = 0.0f;

            Eigen::Vector3d e_center;
            cv::cv2eigen(center_matrix, e_center);

            // 初始位置矩阵SE3 平移位置
            // 世界坐标系中的位置
            Sophus::SE3 center_pose(Eigen::Matrix3d::Identity(), // 世界坐标系下的三维坐标
                                    e_center);

            // 对于rotation 转置 == 求逆
            Eigen::Matrix3d rotation = camera_to_world.rotation_matrix().inverse();

            // world_to_camera 世界坐标系在相机坐标系的位置
            Sophus::SE3 world_to_camera(rotation, Eigen::Vector3d(-camera_to_world.translation()[0],
                                                                  -camera_to_world.translation()[1],
                                                                  -camera_to_world.translation()[2]));

            // 注意在相机坐标系中进行了转换，所以注意切换坐标系回去

            // 得到相机坐标系中的前哨站中心三维坐标
            Sophus::SE3 center_in_camera = world_to_camera * center_pose;

            //
            // DLOG(INFO) << camera_to_world.translation() << std::endl;

            Mat T_matrix;
            cv::eigen2cv(center_in_camera.translation(), T_matrix);

            double temp = T_matrix.ptr<double>(0)[2];                // x = x
            T_matrix.ptr<double>(0)[2] = T_matrix.ptr<double>(0)[1]; // z = y
            T_matrix.ptr<double>(0)[1] = -temp;                      // y = -z

            // DLOG(INFO) << "T_Matrix: " << T_matrix <<std::endl;

            // 利用相机内参转换
            // 相机内参 x 相机坐标系下的坐标
            center_matrix = camera_matrix * T_matrix;
            // std::cout << "---------" <<std::endl;
            // std::cout << center_matrix <<std::endl;

            // uv 坐标
            double u = center_matrix.at<double>(0, 0) / center_matrix.at<double>(2, 0);
            double v = center_matrix.at<double>(1, 0) / center_matrix.at<double>(2, 0);

            // DLOG(WARNING)<< "二维像素坐标" << u <<" , " <<v << std::endl;
            // std::cout<<" Pixel: " << u <<" "<< v << std::endl;
            return Point2f(u, v);
        }
        return center_pixel;
    }

    // TODO 未测
    // 半速前哨站模式
    // Point2f PoseSolver::halfoutpostMode(vector<ArmorBlob> &armors, double delta_t, const SerialPortData &imu_data, SerialPort *SerialPort_)
    // {
    // // 前哨站的中心点
    // Point2f center_pixel = Point2f(0, 0);

    // // 没有装甲板，返回
    // if (armors.size() < 1)
    //     return center_pixel;

    // // 解算装甲板的位姿
    // for (ArmorBlob &a : armors)
    //     solveArmor(a, imu_data);

    // // 候选装甲板
    // vector<ArmorBlob> candidates;

    // // DLOG(INFO) << "state: OUTPOST";

    // // 击打前哨站
    // for (auto &a : armors)
    // {
    //     if (a._class == 7) // 前哨战
    //     {
    //         // 对于5m来看的
    //         // a.x *= 6.2 / 5; // 这个是估计出来的, 后面再说, 乘一个小倍数 6.2/5
    //         // a.y *= 6.2 / 5;
    //         // a.z *= 6.2 / 5;
    //         candidates.push_back(a);
    //     }
    // }

    // // 超过一个候选装甲板
    // if (candidates.size() < 1)
    // {
    //     last_armor = armors.at(0); // 更新上一帧的装甲板 last_armor
    //     return center_pixel;
    // }

    // // DLOG(WARNING)<<"size: "<<candidates.size()<<std::endl;

    // right_clicked = imu_data.right_clicked;
    // if (last_right_clicked == 0 && right_clicked == 1)
    //     first = true;

    // // 第一次右击
    // if (first)
    // {
    //     outpost.clear();
    //     first = false;
    // }
    // last_right_clicked = right_clicked;
    // // DLOG(INFO) << " right clicked: " << right_clicked;

    // // 识别到就更新
    // // 更新前哨站中心
    // for (const auto &a : candidates)
    // {
    //     // 未满就更新
    //     if (outpost.getSize() < outpost.size()) // 当没满的时候重新对准
    //         outpost.update({a.x, a.y, a.z});

    //     // 当装甲板距离中心一定距离之后才会更新
    //     else
    //     {
    //         // center = outpost.getMetric() / outpost.size();
    //         // std::cout<<"full"<<std::endl;

    //         // 约束更新的数据在一定范围内, 两侧的装甲板的中心测量不准确
    //         // 当坐标在一定范围内才会更新，这就表示了距离在0.4m内部时，才会去更新前哨站的中心
    //         // 会使得中心的抖动不会那么明显，从而云台比较稳定

    //         // 在靠近中心一定范围内才会取更新中心
    //         if (sqrt((a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y) + (a.z - center.z) * (a.z - center.z)) < 0.15)
    //         {
    //             DLOG(WARNING) << sqrt((a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y) + (a.z - center.z) * (a.z - center.z)) << std::endl;
    //             outpost.update({a.x, a.y, a.z});
    //         }
    //     }
    // }

    // // 更新roll
    // // if (abs(imu_data.roll - roll.getMetric() / roll.getSize()) < 1000)
    // // {
    // //     roll.update(imu_data.roll);
    // // }

    // // 旋转速度
    // double w = 0.2 * 360;

    // // DLOG(INFO) << "                                    center x: " << center.x << " y: " << center.y << " z: " << center.z << " roll: " << roll.getMetric() / roll.getSize();

    // // 当发射的子弹没有打出去之前，不更新 当前的位姿，也就是不移动当前的头
    // // 等待子弹打出去之后，再移动，一旦锁定之后，就不再移动，知道打出去这发子弹
    // // 当子弹发出 的一个时间范围内，不更新云台的数据，也就是保持在原位， 超过这个时间戳之后再更新当前的位姿
    // if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - send_shoot_time).count() < ((120 / w) * 1.3 * 1000))
    // {
    //     // 处于延时等待, 此时云台不再跟随，直到子弹打出去
    //     // DLOG(INFO) << "waiting shoot" << std::endl;
    // }
    // else
    // {
    //     // 当循环数组没满时，选择距离上一次最近的装甲板加入作为选中的
    //     // 根据距离排序
    //     sort(candidates.begin(), candidates.end(), [&](const ArmorBlob &a, const ArmorBlob &b)
    //          { return calcDiff(a, last_armor) < calcDiff(b, last_armor); });

    //     // 选择距离上一次装甲板最近
    //     armor = candidates.at(0);
    //     center.x = armor.x;
    //     center.y = armor.y;
    //     center.z = armor.z;

    //     // 满足后，可以认为中心已经找到，然后就进行处理
    //     if (outpost.getSize() == outpost.size())
    //     {
    //         // 估计前哨站中心
    //         center = outpost.getMetric() / outpost.size();

    //         // 距离
    //         double distance = sqrt(center.x * center.x + center.y * center.y + center.z * center.z);

    //         DLOG(WARNING) << "distance:" << distance << std::endl;

    //         // 发弹延时0.01
    //         double time = distance / speed;

    //         SerialParam::send_data.shootStatus = 0;

    //         // 选择装甲板准备进行 延时击打
    //         for (const auto &a : candidates)
    //         {
    //             // DLOG(INFO) << "center diff: " << abs(a.x - center.x);
    //             // 只考虑xz，也就是水平方向的距离, 距离一定时
    //             double armor_outpost_distance = sqrt((center.x - a.x) * (center.x - a.x) + (center.z - a.z) * (center.z - a.z));

    //             // if (GlobalParam::DEBUG_MODE)
    //             // centerFrame.distance = (float)armor_outpost_distance;

    //             // 最早使用的击打标准
    //             // if (abs(a.x - center.x) < 0.02 && abs(a.y - center.y) < 0.05)   // x坐标小于20cm, y坐标小于 50cm

    //             // DLOG(ERROR) << armor_outpost_distance << std::endl;
    //             // bool flag = false;
    //             // Point2f armor_pixel_center = Point2f(a.rect.x + a.rect.width / 2, a.rect.y + armor_height / 2);
    //             // double pixel_distance = sqrt((armor_pixel_center.x - u) * (armor_pixel_center.x - u) + (armor_pixel_center.y - v) * (armor_pixel_center.y - v));
    //             // 当中心点距离与重投影中心距离小于宽度的1/10时，进行击打
    //             // DLOG(WARNING) << armor_pixel_center.x - u<<std::endl;
    //             // DLOG(WARNING) << (armor_pixel_center.y-v)<<std::endl;
    //             // DLOG(WARNING) << pixel_distance<<std::endl;
    //             // DLOG(WARNING) << a.rect.width / 5<<std::endl;

    //             // flag = pixel_distance < a.rect.width * OutpostParam::center_ratio;

    //             // 修改后使用的击打标准
    //             if (armor_outpost_distance < 0.05) // 在10cm的范围内
    //             // if (flag)
    //             {
    //                 // 更新发射子弹时间
    //                 send_shoot_time = std::chrono::steady_clock::now();

    //                 // 休眠时间
    //                 double tmp_time = (120 / w - time) * 1000 + OutpostParam::time_bias; // ms 还要考虑到机械这方面控制的时间
    //                 DLOG(WARNING) << (120 / w) * 1000 << std::endl;
    //                 DLOG(WARNING) << time * 1000 << std::endl;
    //                 DLOG(WARNING) << OutpostParam::time_bias << std::endl;
    //                 DLOG(WARNING) << "wait_time: " << tmp_time << std::endl;
    //                 thread([this, a, tmp_time, w, SerialPort_]()
    //                        {
    //                     int sleep_time = tmp_time;
    //                     // int sleep_time = (120/w-time)*1000 - 0.01; // ms
    //                     std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    //                     SerialParam::send_data.shootStatus = 1;
    //                     SerialPort_->writeData(&SerialParam::send_data);
    //                     DLOG(INFO) << "                shoot"; })
    //                     .detach();
    //                 break;
    //             }
    //         }
    //     }

    //     // 但是没调好，暂时不使用
    //     if (abs(roll.getMetric() / roll.getSize()) > 700)
    //     {
    //         // TODO roll偏差较大，认为此时在斜坡上
    //         // FIXME 调整弹道
    //         // double offset = predictor->fitTrajectory(center, speed, true);              // true表示使用roll的数据
    //         double bias = 0;
    //         // SerialParam::send_data.yaw = predictor->getYaw(center, delta_t, speed, imu_data, false) + (offset + bias) * sin(roll.getMetric() / roll.getSize() / 100 / 180 * 3.14) - 100;
    //         // SerialParam::send_data.pitch = predictor->fitTrajectory(center, speed, false) - offset + (offset + bias) * cos(-roll.getMetric() / roll.getSize() / 100 / 180 * 3.14);
    //         SerialParam::send_data.yaw = predictor->getYaw(center, delta_t, speed, imu_data, false);
    //         SerialParam::send_data.pitch = predictor->fitTrajectory(center, speed, false);
    //     }
    //     else

    //     {

    //         // 解算yaw轴的数据，false表示不需要预测
    //         SerialParam::send_data.yaw = predictor->getYaw(center, delta_t, speed, imu_data, false); // 不使用预测，而是瞄准中心点的位置

    //         // TODO 弹道补偿
    //         // FIXME 下面有一个超参数50，重力补偿得到的，稍微网上抬一点
    //         // SerialParam::send_data.pitch = predictor->fitTrajectory(center, speed, false)+ 50 ;             // false 表示不使用roll的数据
    //         SerialParam::send_data.pitch = predictor->fitTrajectory(center, speed, false) + OutpostParam::pitch_bias; // false 表示不使用roll的数据
    //     }
    // }

    // // 前哨站模式下进行调试
    // // 使用VOFA+ 进行udp通信调试
    // // if (GlobalParam::SOCKET)
    // // {
    // //     centerFrame.x = center.x;
    // //     centerFrame.y = center.y;
    // //     centerFrame.z = center.z;
    // //     centerFrame.distance = sqrt(center.x * center.x + center.y * center.y + center.z * center.z);
    // //     centerFrame.pitch = SerialParam::send_data.pitch;
    // //     centerFrame.yaw = SerialParam::send_data.yaw;
    // //     udpsender->send(centerFrame);
    // // }

    // // DLOG(INFO) << "                                           angle: " << yaw;
    // // DLOG(INFO) << "                                           send yaw: " << SerialParam::send_data.yaw << " send pitch: " << SerialParam::send_data.pitch;
    // // DLOG(INFO) << "                                           recv yaw: " << SerialParam::recv_data.yaw << "  recv pitch: " << SerialParam::recv_data.pitch << "  recv roll: " << imu_data.roll;
    // // DLOG(INFO) << "                                           x: " << armor.x << " y: " << armor.y << " z: " << armor.z;
    // last_armor = armor;

    // // 绘制前哨站中心
    // if (GlobalParam::DEBUG_MODE)
    // {

    //     //  重投影 计算前哨战中心点部分

    //     // 1. 已知： 前哨战中心的三维坐标
    //     //          相机和世界坐标系的转换关系
    //     //          相机内参矩阵
    //     // 2. 未知:  通过逆操作，求出相机坐标系中的前哨战的坐标

    //     // 世界坐标系  de  前哨站中心点
    //     Mat center_matrix = cv::Mat(3, 1, CV_64FC1, cv::Scalar::all(0));
    //     center_matrix.ptr<double>(0)[0] = -center.x;
    //     center_matrix.ptr<double>(1)[0] = -center.y;
    //     center_matrix.ptr<double>(2)[0] = center.z;
    //     // center_matrix.ptr<double>(3)[0] = 1.0f;

    //     // std::cout << center_matrix << std::endl;

    //     // 相机中心在世界坐标系中的坐标
    //     // Mat camera2world = cv::Mat(3, 1, CV_64FC1, cv::Scalar::all(0));
    //     // camera2world.ptr<double>(0)[0] = camera_to_world.translation()[0];
    //     // camera2world.ptr<double>(1)[0] = camera_to_world.translation()[1];
    //     // camera2world.ptr<double>(2)[0] = camera_to_world.translation()[2];
    //     // camera2world.ptr<double>(3)[0] = 0.0f;

    //     Eigen::Vector3d e_center;
    //     cv::cv2eigen(center_matrix, e_center);

    //     // 初始位置矩阵SE3 平移位置
    //     // 世界坐标系中的位置
    //     Sophus::SE3 center_pose(Eigen::Matrix3d::Identity(), // 世界坐标系下的三维坐标
    //                             e_center);

    //     // 对于rotation 转置 == 求逆
    //     Eigen::Matrix3d rotation = camera_to_world.rotation_matrix().inverse();

    //     // world_to_camera 世界坐标系在相机坐标系的位置
    //     Sophus::SE3 world_to_camera(rotation, Eigen::Vector3d(-camera_to_world.translation()[0],
    //                                                           -camera_to_world.translation()[1],
    //                                                           -camera_to_world.translation()[2]));

    //     // 得到相机坐标系中的前哨站中心三维坐标
    //     Sophus::SE3 center_in_camera = world_to_camera * center_pose;

    //     // std::cout << center_matrix <<std::endl;
    //     // std::cout << "----------" << std::endl;
    //     // std::cout << camera_to_world.translation() << std::endl;
    //     // std::cout << center_in_camera.translation() <<std::endl;

    //     Mat T_matrix;
    //     cv::eigen2cv(center_in_camera.translation(), T_matrix);

    //     // 利用相机内参转换
    //     center_matrix = camera_matrix * T_matrix;
    //     // std::cout << "---------" <<std::endl;
    //     // std::cout << center_matrix <<std::endl;

    //     // uv 坐标
    //     double u = center_matrix.at<double>(0, 0) / center_matrix.at<double>(2, 0);
    //     double v = center_matrix.at<double>(1, 0) / center_matrix.at<double>(2, 0);
    //     // std::cout<<" Pixel: " << u <<" "<< v << std::endl;

    //     return Point2f(u, v);
    // }

    // return center_pixel;
    // }

    float PoseSolver::calcDiff(const ArmorBlob &a, const ArmorBlob &b) {
        return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
    }

    // 公用的函数
    void PoseSolver::clearCircle() {
        predictor->resetPredictor();
        outpost.clear();            // 清空中心区域
        outpost_center.clear();
        filtered_pitch.clear();     // 清空pitch稳定
        // thresh.clear();
    }

    void PoseSolver::clearSentinel() {
        sentinel.clear();
    }

    // 瞄准跟踪的装甲板
    int PoseSolver::chooseArmor(const vector<ArmorBlob> &armors) {
        bool has_same_class = false;
        bool has_hero = false;
        bool has_sentry = false;
        bool has_engineer = false;
        for (const auto &a: armors) {
            if (a._class == 6) // 优先选择最高优先级 相当于操作手指定的类别
                return 6;

            if (a._class == 7)
                return 7;
            cout<<a._class<<" "<<last_armor._class<<endl;
            if (a._class == last_armor._class) // 其次选择和上一帧识别到的类别一样的装甲板
                has_same_class = true;

            switch (a._class) {
                case 1:
                    has_hero = true;
                    break;
                case 2:
                    has_engineer = true;
                    break;
                case 3:
                case 4:
                case 5:
                    has_sentry = true;
                    break;
            }
        }

        // 没有操作手指定的目标时
        if (has_same_class)                 // 选择和上一次一样的类别
            return last_armor._class;
        else if (has_hero)                  // 优先选择英雄
            return 1;
        else if (has_sentry)                // 其次选择步兵
        {
            double d = 10;
            int c = 0;

            for (const auto &a: armors)    // 选择距离在 10m 以内的步兵
            {
//                cout<<"d: "<<a.x * a.x + a.z * a.z<<endl;
                double distance = sqrt(a.x * a.x + a.z * a.z + a.y * a.y);
                if ((a._class == 3 || a._class == 4 || a._class == 5) && d > distance) {
                    d = a.x * a.x + a.z * a.z;
                    c = a._class;
                }
            }
            return c;
        } else if (has_engineer)              // 最后再选择工程
            return 2;
        return 0;
    }

    Point2f PoseSolver::reproject(Point3f center) {
        // 将center转成eigen
        Eigen::Vector3d e_center;
        e_center << center.x, center.y, center.z;

        e_center = armor_to_world.inverse() * e_center;
        vector<Point3f> center3D;
        center3D.push_back(Point3f(e_center[0], e_center[1], e_center[2]));
        vector<cv::Point2f> reprojected_points;
        cv::projectPoints(center3D, rvec, tvec, camera_matrix, distortion_coefficients, reprojected_points);

        return reprojected_points[0];
    }



    /**
     * @brief 光束法平差 这一部分的作用，就是更新了装甲板的宽高，并且重新得到了装甲板在相机坐标系下的位姿
     *
     * @param a_b 装甲板的  宽/2 高/2
     * @param points_2d 装甲板四个角点坐标
     * @param K 相机内参矩阵
     * @param R solvePnpGeneric得到的旋转矩阵
     * @param t solvePnPGeneric得到的平移向量
     * @return Eigen::Isometry3d
     */
    // Eigen::Isometry3d PoseSolver::bundleAdjustment(Eigen::Vector2d &a_b, Eigen::Matrix<double, 4, 2> &points_2d, const Mat &K, Mat &R, Mat &t)
    // {
    //     // 初始化g2o

    //     // 位姿的维度为6，路标的维度为2
    //     // pose维度为6，landmark维度为2
    //     typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block; // pose 维度为 6, landmark 维度为 2
    //     // linearsolver 使用 Eigen库实现矩阵求逆来解方程
    //     std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverEigen<Block::PoseMatrixType>()); // 方程求解器
    //     std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));                                      // 矩阵块求解器
    //     // 初始化一个求解器
    //     g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    //     g2o::SparseOptimizer optimizer;
    //     optimizer.setAlgorithm(solver);

    //     // vertex
    //     // 将四个点的坐标分别设置为g2o中的顶点，即为优化的变量，坐标变量为初始值
    //     g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap(); // camera pose

    //     Eigen::Matrix3d R_mat;
    //     // 旋转矩阵
    //     R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    //         R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    //         R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

    //     // pose 初始化
    //     pose->setId(0);
    //     // 旋转矩阵，平移向量组成一个Isometry3d对象
    //     // 初始化相机位姿，旋转矩阵、平移向量
    //     pose->setEstimate(g2o::SE3Quat(
    //         R_mat,
    //         Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))));
    //     optimizer.addVertex(pose);

    //     // 定义右下角坐标 顶点并设置优化变量
    //     VertexAB *ab = new VertexAB();
    //     ab->setId(1);
    //     ab->setEstimate(a_b);
    //     ab->setMarginalized(true);
    //     optimizer.addVertex(ab);

    //     // OrthogonalEdge
    //     // 用四个顶点的2D坐标和顶点a,b 在g2o中初始化一套OrthogonalEdge
    //     // 用于约束每条射线在相机坐标系下的交点应该都位于四个三角形的内部，即最小化误差项
    //     // 构造误差项：计算4个点到光心的射线与装甲板角点构成的三角形内部误差
    //     OrthogonalEdge *edge_ = new OrthogonalEdge();
    //     edge_->setId(1);
    //     edge_->setVertex(0, ab);
    //     edge_->setVertex(1, pose);
    //     edge_->setMeasurement(points_2d);
    //     edge_->setInformation(Eigen::Matrix<double, 8, 8>::Identity());
    //     optimizer.addEdge(edge_);

    //     // 优化器设置以及运行
    //     std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    //     optimizer.setVerbose(true);
    //     optimizer.initializeOptimization();
    //     // 进行优化
    //     optimizer.optimize(5);

    //     // 记录优化所需要的时间
    //     std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    //     std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //     cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    //     // 优化后的结果
    //     cout << endl
    //          << "after optimization:" << endl;
    //     cout << "T=" << endl
    //          << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
    //     cout << "a=" << ab->estimate().matrix()[0] << endl;
    //     cout << "b=" << ab->estimate().matrix()[1] << endl;

    //     // 优化后的结果： 宽/高 比例
    //     float length_rate_height = ab->estimate().matrix()[1] / ab->estimate().matrix()[0];
    //     // 优化后的高
    //     float height_now = length_of_small * length_rate_height;

    //     // 更新装甲板的高度
    //     height_of_small = height_now;

    //     // ba_points
    //     std::vector<cv::Point3f>
    //         ba_ponits{{-length_of_small, -height_now, 0}, {length_of_small, -height_now, 0}, {length_of_small, height_now, 0}, {-length_of_small, height_now, 0}};

    //     // 将优化后的四点重新写回
    //     // 覆盖掉优化后的
    //     rewrite_3d_points(ba_ponits);

    //     // 返回优化后的 相机 位姿
    //     return Eigen::Isometry3d(pose->estimate());
    // }

    /**
     * @brief 覆盖掉原来的装甲板坐标系的坐标
     *
     * @param points_3d_after_ba 优化后的装甲板坐标系四个角点的坐标
     */
    // void PoseSolver::rewrite_3d_points(std::vector<cv::Point3f> &points_3d_after_ba)
    // {
    //     points_3d = points_3d_after_ba;
    // }

    /**
     * @brief
     *
     * @param armor 指定的要解算位姿的装甲板
     * @param imu_data 当前的imu数据，包括pitch、yaw、roll，都是相对于世界坐标系的数据
     *                 pitch、roll数据是根据实际得到的，yaw轴数据是相当于初始上电时的基准来的
     *                 pitch 当前云台与水平面的夹角
     */
    // void PoseSolver::solveArmorBA(ArmorBlob &armor, const SerialPortData &imu_data)
    // {

    //     // 装甲板的位姿
    //     const static Sophus::SE3 armor_pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0));

    //     // TODO: 更新识别大小装甲板的规则
    //     // 最新的装甲板编号中 1 是英雄， 最新规则中只有英雄和大基地是大装甲板
    //     // FIXME 对于平衡步兵要单独增加判断
    //     if (armor._class == 1)
    //     { // large armor
    //         points_3d = points_large_3d;
    //         // halflength_of_armor = 0.1125f;              // 装甲板
    //         // halfheight_of_armor = 0.0275f;
    //     }
    //     else
    //     { // small armor
    //         // 假定就是
    //         points_3d = points_small_3d;
    //     }

    //     // pnp 解算，并计算重投影误差， SOLVEPNP_IPPE只会返回两组解
    //     solvePnPGeneric(points_3d, armor.corners, camera_matrix, distortion_coefficients,
    //                     rvecs, tvecs, false, cv::SOLVEPNP_IPPE, noArray(), noArray(), reproject_error);

    //     // TODO 进行 BA 优化
    //     // 返回的第一组解
    //     cv::Mat R1;
    //     cv::Rodrigues(rvecs[0], R1); // 旋转矢量转换为旋转矩阵
    //     Eigen::Matrix3d Rotate_M1 = Eigen::Matrix3d::Identity();
    //     cv::cv2eigen(R1, Rotate_M1);

    //     // 返回的第二组解
    //     cv::Mat R2;
    //     cv::Rodrigues(rvecs[1], R2); // 旋转矢量转换为旋转矩阵
    //     Eigen::Matrix3d Rotate_M2 = Eigen::Matrix3d::Identity();
    //     cv::cv2eigen(R2, Rotate_M2);

    //     // 四元素
    //     Eigen::Quaterniond q4_1;
    //     Eigen::Isometry3d T_;

    //     // detector检测出来的四个角点
    //     Eigen::Matrix<double, 4, 2> points_2;

    //     // 四个角点
    //     // 0 1
    //     // 3 2
    //     points_2(0, 0) = armor.corners[0].x;
    //     points_2(0, 1) = armor.corners[0].y;
    //     points_2(1, 0) = armor.corners[1].x;
    //     points_2(1, 1) = armor.corners[1].y;
    //     points_2(2, 0) = armor.corners[2].x;
    //     points_2(2, 1) = armor.corners[2].y;
    //     points_2(3, 0) = armor.corners[3].x;
    //     points_2(3, 1) = armor.corners[3].y;

    //     // 装甲板的宽/2 高/2
    //     Eigen::Vector2d two_ = Eigen::Vector2d{-points_3d[0].x, -points_3d[0].y};

    //     //
    //     //          装甲板的宽/2,高/2  四点坐标  相机内参矩阵  旋转矩阵 平移向量  （第一组解）
    //     T_ = bundleAdjustment(two_, points_2, camera_matrix, R1, tvecs[0]);
    //     // BA优化

    //     std::cout << "BA优化完成" << std::endl;

    //     // 优化后的旋转矩阵
    //     q4_1 = Eigen::Quaterniond(T_.rotation().matrix());

    //     // 平移向量, 第一组解是z轴相反的
    //     Eigen::Vector3d translate1(tvecs[0].ptr<double>(0)[0], tvecs[0].ptr<double>(0)[1], -tvecs[0].ptr<double>(0)[2]);

    //     // 第二组解的旋转矩阵
    //     // Eigen::Quaterniond q4_2 = Eigen::Quaterniond(Rotate_M2);
    //     // DLOG(WARNING) << "reproject_error" << reproject_error << std::endl;

    //     // 对于优化后的结果进行滤波
    //     // 重投影误差
    //     // if ((reproject_error[1] / reproject_error[0]) > 1.5 /*&&(reproject_error[1]+reproject_error[2]>4)*/)
    //     // {
    //     //     // reproject_error[0] 比较小，也就是选取的这部分的重投影误差小，就继续使用
    //     //     // DLOG(WARNING) << "reproject_error[1] / reproject_error[0] > 1.5"<<std::endl;
    //     //     if (angle_fliter->dont_believe_measure)
    //     //     {
    //     //         // DLOG(WARNING) << "dont_believe_mesaure" << std::endl;
    //     //         angle_fliter->setMeasurementNoise(0.0001, 0.0001, 0.0001, 0.0001); // 调回测量噪声
    //     //         angle_fliter->dont_believe_measure = false;
    //     //     }
    //     //     // 获取滤波后的滤波后的数据，后验状态估计
    //     //     angle_state_vector = angle_fliter->runKalman(q4_1, delta_t);
    //     //     // DLOG(WARNING)<<"q4_1 "<<q4_1<<" delta_t"<<delta_t<<std::endl;
    //     //     // DLOG(WARNING)<<"angle_state_vector "<<angle_state_vector<<std::endl;

    //     // }
    //     // else
    //     // {
    //     //     // DLOG(ERROR) << "reproject_error" << reproject_error << std::endl;
    //     //     angle_fliter->setMeasurementNoise(0.001, 0.001, 0.001, 0.001); // 调大测量噪声
    //     //     angle_state_vector = angle_fliter->runKalman(q4_1, delta_t);
    //     //     angle_fliter->dont_believe_measure = true;
    //     //     // angle_fliter->if_first_jump=true;
    //     // }

    //     // 旋转矩阵
    //     // Eigen::Matrix3d rotation_matrix3;
    //     // Quaterniond qqq(angle_state_vector[0], angle_state_vector[2], angle_state_vector[4], angle_state_vector[6]);
    //     // rotation_matrix3 = qqq.toRotationMatrix();

    //     // 旋转矩阵、平移向量（优化后的旋转矩阵、第一组解的平移向量）
    //     armor_to_camera = Sophus::SE3(T_.rotation(), translate1);
    //     // armor_to_camera = Sophus::SE3(rotation_matrix3, translate1) * armor_pose;

    //     // cv2eigen(tvec, e_T);
    //     // // 旋转向量转旋转矩阵
    //     // Rodrigues(rvec, m_R);
    //     // cv2eigen(m_R, e_R);

    //     // 将 armor_pose 从装甲板坐标系转换成相机坐标系
    //     // armor_to_camera = Sophus::SE3(e_R, e_T) * armor_pose;
    //     // 相机坐标系坐标 转换成 世界坐标系
    //     camera_to_world = gimbal_to_world * camera_to_gimbal;
    //     // 装甲板坐标 转换到 云台坐标系
    //     armor_to_gimbal = camera_to_gimbal * armor_to_camera;
    //     // 装甲板坐标 转换到 世界坐标系
    //     armor_to_world = camera_to_world * armor_to_camera;

    //     // 得到当前的yaw轴数据      角度转成弧度
    //     float rcv_yaw = imu_data.yaw / 100.0f * M_PI / 180.0f;

    //     // 得到当前的pitch轴数据    角度转弧度
    //     float rcv_pitch = imu_data.pitch / 100.0f * M_PI / 180.0f;

    //     // roll轴的数据
    //     // float rcv_roll = roll.getMetric() / roll.getSize();
    //     // TODO
    //     float rcv_roll = imu_data.roll / 100.0f * M_PI / 180.0f;

    //     // 更新当前的imu数据, 更新云台 -> 世界坐标系的转化矩阵
    //     setimu(rcv_pitch, rcv_yaw, rcv_roll);

    //     // DLOG(INFO) << "                              corners: " << armor.corners;

    //     // double z = atan2(m_R.at<double>(1, 0), m_R.at<double>(0, 0));

    //     // TODO 后续继续理解这个yaw值，这个yaw求的是装甲板的yaw方向上的倾斜角度
    //     // m_R 装甲板坐标系对相机坐标系的yaw偏角
    //     // FIXME 还要进一步阅读理解 yaw轴的偏差角度

    //     // TODO 更新旋转矩阵，要重新测试，获取装甲板对世界坐标系的
    //     Eigen::Matrix3d rotation_matrix = armor_to_camera.rotation_matrix();
    //     cv::eigen2cv(rotation_matrix, m_R);

    //     // 角度
    //     yaw = atan2(-m_R.at<double>(2, 0), sqrt(pow(m_R.at<double>(2, 0), 2) + pow(m_R.at<double>(2, 2), 2))) / M_PI * 180;

    //     DLOG(INFO) << "armor_angle" << yaw;
    //     // double x = atan2(m_R.at<double>(2, 1), m_R.at<double>(2, 2));
    //     // DLOG(INFO) << "z: " << z << " y: " << yaw << " x: " << x;

    //     // 装甲板在云台坐标系下的坐标
    //     // const auto &a = armor_to_gimbal;
    //     // Point3d t = {a.translation()[0],
    //     //              a.translation()[1],
    //     //              a.translation()[2]};

    //     // DLOG(INFO) << t;

    //     // 装甲板在世界坐标系下的坐标
    //     // 得到yaw轴的坐标
    //     Point3d trans = {-armor_to_world.translation()[0], // x
    //                      -armor_to_world.translation()[1], // y
    //                      armor_to_world.translation()[2]}; // z

    //     // 确定一下pitch、yaw、roll和rotation() 坐标的对应关系

    //     // 装甲板的角度
    //     armor.angle = yaw;

    //     // 世界坐标系中的坐标
    //     armor.x = trans.x;
    //     armor.y = trans.y;
    //     armor.z = trans.z;

    //     // 使用VOFA+ 进行udp通信调试
    //     if (GlobalParam::SOCKET)
    //     {
    //         outpostPoseDataFrame.x = armor.x;
    //         outpostPoseDataFrame.y = armor.y;
    //         outpostPoseDataFrame.z = armor.z;
    //         outpostPoseDataFrame.pitch = armor.angle;
    //         // outpostPoseDataFrame.yaw = imu_data.yaw;

    //         // 原始pnp解算结果
    //         const vector<Point3f> points_small = {Point3f(-0.675f, -0.0275f, 0.f),
    //                                               Point3f(0.0675f, -0.0275f, 0.f),
    //                                               Point3f(0.0675f, 0.0275f, 0.f),
    //                                               Point3f(-0.0675f, 0.0275f, 0.f)};

    //         // BA优化前的yaw angle
    //         solvePnP(points_small, armor.corners, camera_matrix, distortion_coefficients,
    //                  rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
    //         cv2eigen(tvec, e_T);

    //         // 旋转向量转旋转矩阵
    //         Rodrigues(rvec, m_R);
    //         cv2eigen(m_R, e_R);
    //         yaw = atan2(-m_R.at<double>(2, 0), sqrt(pow(m_R.at<double>(2, 0), 2) + pow(m_R.at<double>(2, 2), 2))) / M_PI * 180;

    //         outpostPoseDataFrame.yaw = yaw;
    //         // outpostPoseDataFrame.yaw = armor.angle;
    //         // outpostPoseDataFrame.roll = ;
    //         udpsender->send(outpostPoseDataFrame);
    //     }
    // }

    // chatgpt提供的位姿BA优化代码
    // 定义ReprojectionError类
    // struct ReprojectionError
    // {
    //     ReprojectionError(cv::Point2f observed) : observed_(observed) {}

    //     template <typename T>
    //     bool operator()(const T *const camera, const T *const point3D, T *residuals) const
    //     {
    //         // 将相机和3D点参数转换为相机矩阵和变换矩阵
    //         T R[9];
    //         ceres::AngleAxisToRotationMatrix(camera, R);
    //         T t[3] = {camera[3], camera[4], camera[5]};
    //         T P[3];
    //         ceres::AngleAxisRotatePoint(R, point3D, P);
    //         P[0] += t[0];
    //         P[1] += t[1];
    //         P[2] += t[2];
    //         T xp = P[0] / P[2];
    //         T yp = P[1] / P[2];
    //         // 计算重投影误差
    //         residuals[0] = xp - T(observed_.x);
    //         residuals[1] = yp - T(observed_.y);
    //         return true;
    //     }

    //     cv::Point2f observed_;
    // };

    // void PoseSolver::bundleAdjustment(const std::vector<cv::Point3f>& points3D,
    //                   const std::vector<cv::Point2f>& points2D,
    //                   const cv::Mat& camera_matrix,
    //                   const cv::Mat& distortion_coefficients,
    //                   cv::Mat& rvec,
    //                   cv::Mat& tvec) {

    // // 定义Ceres问题
    // ceres::Problem problem;

    // // 定义相机内参和畸变系数
    // double* cm = const_cast<double*>(camera_matrix.ptr<double>());
    // double* dc = const_cast<double*>(distortion_coefficients.ptr<double>());

    // // 定义相机的初始姿态和位置
    // double* r = rvec.ptr<double>();
    // double* t = tvec.ptr<double>();

    // // 定义3D点
    // std::vector<double*> points3DPtrs;
    // for (auto& p : points3D) {
    //     auto point3D_ptr = std::make_unique<double[]>(3);
    //     std::copy_n(reinterpret_cast<const double*>(&p.x), 3, point3D_ptr.get());
    //     points3DPtrs.emplace_back(std::move(point3D_ptr));
    // }

    // // 配置Ceres Solver选项
    // ceres::Solver::Options options;
    // options.max_num_iterations = 100;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.minimizer_progress_to_stdout = true;

    // // 使用多组随机初始值运行BA算法，以获得更准确的结果
    // ceres::Solver::Summary summary;
    // for (int i = 0; i < 10; i++) {
    //     // 随机生成相机的初始姿态和位置
    //     double camera[6] = {r[0] + rand() * 0.1 / RAND_MAX,
    //                         r[1] + rand() * 0.1 / RAND_MAX,
    //                         r[2] + rand() * 0.1 / RAND_MAX,
    //                         t[0] + rand() * 0.1 / RAND_MAX,
    //                         t[1] + rand() * 0.1 / RAND_MAX,
    //                         t[2] + rand() * 0.1 / RAND_MAX};
    //     // 添加残差项
    //     for (size_t i = 0; i < points2D.size(); ++i) {
    //         ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(points2D[i]));
    //         problem.AddResidualBlock(cost_function, nullptr, camera, points3DPtrs[i]);
    //         // problem.AddResidualBlock(cost_function.get(), nullptr, camera, points3DPtrs[i].get());
    //     }

    //     // 解决问题
    //     ceres::Solve(options, &problem, &summary);

    //     // 如果优化成功，则更新相机的姿态和位置
    //     if (summary.IsSolutionUsable()) {
    //         r[0] = camera[0];
    //         r[1] = camera[1];
    //         r[2] = camera[2];
    //         t[0] = camera[3];
    //         t[1] = camera[4];
    //         t[2] = camera[5];
    //     }
    // }
    // }
}