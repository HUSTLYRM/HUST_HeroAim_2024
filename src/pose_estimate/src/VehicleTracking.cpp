// #include "VehicleTracking.h"
// namespace ly
// {

// #define PI 3.1415926

//     double x = 0;
//     double y = 1;
//     double z = 0;
//     double yaw = -PI / 2;
//     VehicleTracking::VehicleTracking()
//     {
//         extended_kalman_filter = new ExtendedKalman<double, 9, 16>();
//         is_kalman_init = false;

//         // 观测矩阵一直保持不变
//         setObservationMatrix();
//         setQandRMatrix();

//         // window = cv::viz::Viz3d("window");
//         // coordinate_system = cv::viz::WCoordinateSystem(1.0);
//     }

//     VehicleTracking::~VehicleTracking()
//     {
//         delete extended_kalman_filter;
//     }

//     //    NOT_GET_TARGET = 0,       // 未获得目标
//     //    CONTINOUS_GET_TARGET = 1, // 连续获得目标
//     //    LOST_BUMP = 2,            // 缓冲阶段
//     //    DETECT_BUMP = 3           // 进入连续识别状态的缓冲

//     Eigen::Vector3d VehicleTracking::predictVehicleState(const Sophus::SE3 &armor_pose, const Sophus::SE3 &armor_pose_sec, bool is_get_second_armor, int &detect_mode, float shoot_time, float frame_delta_t, float yaw_gimbal)
//     {

//         std::cout << "time: " << frame_delta_t << std::endl;
//         frame_delta_t = 5;
//         Eigen::Vector3d predict_point;
//         Eigen::Vector4d measure, measure_sec;

//         Eigen::Matrix<double, 9, 1> whole_car_state;

//         measure = PoseToMeasurement(armor_pose);

//         measure[3] = measure[3] > PI / 2 ? measure[3] - PI : measure[3];
//         yaw_send = measure[3];

//         if (is_get_second_armor)
//         {
//             measure_sec = PoseToMeasurement(armor_pose_sec);
//             // 根据两个measure的x大小进行重排,measure在左,measure_sec在右
//             if (measure[0] > measure_sec[0])
//             {
//                 std::swap(measure, measure_sec);
//             }
//         }

//         // z = 0;
//         // yaw += 0.5 * frame_delta_t / 1000;
//         // if (-PI / 2 <= yaw <= 0)
//         // {
//         // x += 0.5 * frame_delta_t / 1000;
//         // y += 0.5 * frame_delta_t / 1000;
//         // }
//         // else if (0 < yaw <= -PI / 2)
//         // {

//         //     x += 1 * frame_delta_t / 1000;
//         //     y += 1 * frame_delta_t / 1000;
//         // }
//         // else if (yaw > PI / 2)
//         // {
//         //     yaw = -PI / 2;
//         //     x = 0;
//         //     y = 1;
//         // }
//         // measure[0] = x;
//         // measure[1] = y;
//         // measure[2] = z;
//         // measure[3] = 0;

//         DLOG(INFO)
//             << "YYYYYYYYYYYYYYYYYYYYYYYY" << measure[3] << std::endl;
//         if (!is_kalman_init)
//         {
//             // init时，生成装甲板的序列基准
//             setArmorSerialBase(measure, measure_sec, is_get_second_armor);

//             is_kalman_init = true;
//             rebootKalman();
//         }
//         else
//         {
//             setUpdateTime(frame_delta_t);
//             if (detect_mode == 2) // LOST_BUMP,进行无量测更新
//             {
//                 whole_car_state = runKalmanWithoutMeasure();
//             }
//             else // CONTINOUS_GET_TARGET,正常预测更新
//             {
//                 // 根据armorSerialBase确认目前观测到装甲板的序号
//                 Eigen::Matrix<double, 16, 1> whole_car_measure = ComfirmArmorSerialNow(measure, measure_sec, is_get_second_armor);
//                 whole_car_state = runKalmanWithMeasure(whole_car_measure);
//             }
//         }
//         DLOG(WARNING) << "WHOLE CAR STATE" << std::endl
//                       << whole_car_state << std::endl;

//         predict_point = getPredictPoint(whole_car_state, shoot_time);

//         DLOG(WARNING) << "PREDICT POINT" << std::endl
//                       << predict_point << std::endl;

//         // vizDebug();

//         return predict_point;
//     }

//     void VehicleTracking::resetTracker()
//     {
//         is_kalman_init = false;
//     }

//     void VehicleTracking::rebootKalman()
//     {
//         extended_kalman_filter->post_state = vehicle_state;
//         extended_kalman_filter->P_post = Eigen::Matrix<double, 9, 9>::Identity();
//     }

//     Eigen::Matrix<double, 9, 1> VehicleTracking::runKalmanWithoutMeasure()
//     {
//         setTransitionMatrix();
//         Eigen::Matrix<double, 9, 1> state = extended_kalman_filter->predict();
//         return state;
//     }

//     Eigen::Matrix<double, 9, 1> VehicleTracking::runKalmanWithMeasure(Eigen::Matrix<double, 16, 1> &measure)
//     {
//         setTransitionMatrix();
//         extended_kalman_filter->predict();
//         Eigen::Matrix<double, 9, 1> state = extended_kalman_filter->update(measure);
//         updateArmorSerialBase();
//         return state;
//     }

//     Eigen::Vector3d VehicleTracking::getPredictPoint(const Eigen::Matrix<double, 9, 1> vehicle_state, float &shoot_t)
//     {
//         std::vector<Eigen::Vector4d> armor_serial_temp = armor_serial_base;
//         // 对每个armor用vehicle_state做更新,找出abs(yaw)最小的
//         for (int i = 0; i < 4; i++)
//         {
//             armor_serial_temp[i][3] = armor_serial_temp[i][3] + vehicle_state[7] * shoot_t;
//         }
//         // std::sort(armor_serial_temp.begin(), armor_serial_temp.end(), [](const Eigen::Vector4d &a, const Eigen::Vector4d &b)
//         //           { return std::abs(a[3]) < std::abs(b[3]); });
//         speed_vector.clear();
//         speed_vector.push_back(vehicle_state[4]);
//         speed_vector.push_back(vehicle_state[5]);
//         speed_vector.push_back(vehicle_state[7]);
//         speed_vector.push_back(shoot_t);
//         Eigen::Vector3d hit_point;
//         hit_point[0] = armor_serial_temp[0][0] + vehicle_state[4] * shoot_t - vehicle_state[8] * vehicle_state[7] * shoot_t * cos(armor_serial_temp[0][3]);
//         hit_point[1] = armor_serial_temp[0][1] + vehicle_state[5] * shoot_t + vehicle_state[8] * vehicle_state[7] * shoot_t * sin(armor_serial_temp[0][3]);
//         hit_point[2] = armor_serial_temp[0][2] + vehicle_state[6] * shoot_t;
//         DLOG(ERROR) << "TTTTTTTTTTTTTTTTT" << shoot_t << std::endl;
//         return hit_point;
//     }

//     // 生成装甲板序列基准
//     void VehicleTracking::setArmorSerialBase(Eigen::Vector4d armor_measure, Eigen::Vector4d armor_measure_sec, bool is_get_second_armor)
//     {

//         Eigen::Vector4d armor_measure_thrid, armor_measure_fourth;

//         armor_serial_base.clear();
//         armor_serial_base.push_back(armor_measure); // 装甲板a
//         if (is_get_second_armor)                    // 获得了第二块装甲板
//         {

//             // 有两块装甲板，可以大致解出半径
//             armor_serial_base.push_back(armor_measure_sec); // 装甲板b
//             double r = (armor_measure[0] - armor_measure_sec[0]) / (sin(armor_measure_sec[3]) - sin(armor_measure[3])) > 0.25 ? 0.25 : (armor_measure[0] - armor_measure_sec[0]) / (sin(armor_measure_sec[3]) - sin(armor_measure[3]));
//             double x_c = armor_measure[0] + r * sin(armor_measure[3]);
//             double y_c = armor_measure[1] + r * cos(armor_measure[3]);

//             armor_measure_thrid[0] = x_c - r * sin(armor_measure[3] - PI);
//             armor_measure_thrid[1] = y_c - r * cos(armor_measure[3] - PI);
//             armor_measure_thrid[2] = armor_measure[2];
//             armor_measure_thrid[3] = armor_measure[3] - PI;

//             armor_measure_thrid[0] = x_c - r * sin(armor_measure_sec[3] - PI);
//             armor_measure_thrid[1] = y_c - r * cos(armor_measure_sec[3] - PI);
//             armor_measure_thrid[2] = armor_measure_sec[2];
//             armor_measure_thrid[3] = armor_measure_sec[3] - PI;

//             armor_serial_base.push_back(armor_measure_thrid);
//             armor_serial_base.push_back(armor_measure_fourth);

//             vehicle_state << x_c, y_c, (armor_measure[2] + armor_measure_sec[2]) / 2, armor_measure[3], 0, 0, 0, 0, r;
//         }
//         else // 没获得第二块装甲板
//         {

//             // 只有一块装甲板,半径按20cm算
//             Eigen::Vector4d armor_sec_temp;
//             armor_sec_temp[0] = armor_measure[0] + 0.2 * (sin(armor_measure[3]) - sin(armor_measure[3] - PI / 2));
//             armor_sec_temp[1] = armor_measure[1] + 0.2 * (cos(armor_measure[3]) - cos(armor_measure[3] - PI / 2));
//             armor_sec_temp[2] = armor_measure[2];
//             armor_sec_temp[3] = armor_measure[3] - PI / 2;
//             armor_measure_sec = armor_sec_temp;
//             armor_serial_base.push_back(armor_measure_sec);

//             armor_measure_thrid[0] = armor_measure[0] + 0.2 * (sin(armor_measure[3]) - sin(armor_measure[3] - PI));
//             armor_measure_thrid[1] = armor_measure[1] + 0.2 * (cos(armor_measure[3]) - cos(armor_measure[3] - PI));
//             armor_measure_thrid[2] = armor_measure[2];
//             armor_measure_thrid[3] = armor_measure[3] - PI;

//             armor_measure_fourth[0] = armor_measure[0] + 0.2 * (sin(armor_measure[3]) - sin(armor_measure[3] + PI / 2));
//             armor_measure_fourth[1] = armor_measure[1] + 0.2 * (cos(armor_measure[3]) - cos(armor_measure[3] + PI / 2));
//             armor_measure_fourth[2] = armor_measure[2];
//             armor_measure_fourth[3] = armor_measure[3] + PI / 2;

//             armor_serial_base.push_back(armor_measure_thrid);
//             armor_serial_base.push_back(armor_measure_fourth);

//             vehicle_state << armor_measure[0] + 0.2 * sin(armor_measure[3]), armor_measure[1] + 0.2 * cos(armor_measure[3]), armor_measure[2], armor_measure[3], 0, 0, 0, 0, 0.2;
//         }
//     }

//     void VehicleTracking::updateArmorSerialBase()
//     {
//         armor_serial_base.clear();
//         Eigen::Vector4d armor_temp;
//         armor_temp = {extended_kalman_filter->pri_state_measure[0], extended_kalman_filter->pri_state_measure[1], extended_kalman_filter->pri_state_measure[2], extended_kalman_filter->pri_state_measure[3]};
//         armor_serial_base.push_back(armor_temp);
//         armor_temp = {extended_kalman_filter->pri_state_measure[4], extended_kalman_filter->pri_state_measure[5], extended_kalman_filter->pri_state_measure[6], extended_kalman_filter->pri_state_measure[7]};
//         armor_serial_base.push_back(armor_temp);
//         armor_temp = {extended_kalman_filter->pri_state_measure[8], extended_kalman_filter->pri_state_measure[9], extended_kalman_filter->pri_state_measure[10], extended_kalman_filter->pri_state_measure[11]};
//         armor_serial_base.push_back(armor_temp);
//         armor_temp = {extended_kalman_filter->pri_state_measure[12], extended_kalman_filter->pri_state_measure[13], extended_kalman_filter->pri_state_measure[14], extended_kalman_filter->pri_state_measure[15]};
//         armor_serial_base.push_back(armor_temp);
//     }

//     // 通过与序列基准对比确定目前的装甲板序号
//     Eigen::Matrix<double, 16, 1> VehicleTracking::ComfirmArmorSerialNow(Eigen::Vector4d &armor_measure, Eigen::Vector4d &armor_measure_sec, bool is_get_second_armor)
//     {

//         if (!is_get_second_armor) // 只观测到一个装甲板
//         {
//             std::vector<std::tuple<int, double>> dists;
//             for (int i = 0; i < 4; i++)
//             {

//                 double diff_x = armor_measure[0] - armor_serial_base[i][0];
//                 double diff_y = armor_measure[1] - armor_serial_base[i][1];
//                 double diff_z = armor_measure[2] - armor_serial_base[i][2];
//                 double diff_yaw = armor_measure[3] - armor_serial_base[i][3];
//                 double dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z); // * (1 + abs(sin(diff_yaw)));

//                 dists.push_back(std::make_tuple(i, dist));
//             }

//             // 按照相差程度从小到大排序
//             std::sort(dists.begin(), dists.end(), [](const auto &lhs, const auto &rhs)
//                       { return std::get<1>(lhs) < std::get<1>(rhs); });

//             int cur_armor_index = std::get<0>(dists[0]);

//             DLOG(WARNING) << "YAW :" << armor_serial_base[cur_armor_index][3] << std::endl;

//             // 让armor_measure的yaw取值从(-PI,PI)延伸到跟整车状态的yaw一个范围(-unlitmit,+unlimit)
//             int vehicle_yaw_region = round((armor_serial_base[cur_armor_index][3] - armor_measure[3]) / (2 * PI));
//             DLOG(ERROR) << "YAW REGION:" << vehicle_yaw_region << std::endl;
//             armor_measure[3] += 2 * PI * vehicle_yaw_region;

//             DLOG(WARNING) << "INDEX:" << cur_armor_index << std::endl;
//             return makeMeasureVector(armor_measure, cur_armor_index);
//         }
//         else // 观测到两个装甲板
//         {
//             std::vector<std::tuple<int, double>> dists;
//             for (int i = 0; i < 4; i++)
//             {
//                 double diff_x = armor_measure[0] - armor_serial_base[i][0];
//                 double diff_y = armor_measure[1] - armor_serial_base[i][1];
//                 double diff_z = armor_measure[2] - armor_serial_base[i][2];
//                 double diff_yaw = armor_measure[3] - armor_serial_base[i][3];
//                 double dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z); //* (1 + abs(sin(diff_yaw)));

//                 int j = i + 1 > 3 ? 0 : i + 1;
//                 diff_x = armor_measure_sec[0] - armor_serial_base[j][0];
//                 diff_y = armor_measure_sec[1] - armor_serial_base[j][1];
//                 diff_z = armor_measure_sec[2] - armor_serial_base[j][2];
//                 diff_yaw = armor_measure_sec[3] - armor_serial_base[j][3];
//                 dist += std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z); // * (1 + abs(sin(diff_yaw)));

//                 dists.push_back(std::make_tuple(i, dist));
//             }

//             // 按照相差程度从小到大排序
//             std::sort(dists.begin(), dists.end(), [](const auto &lhs, const auto &rhs)
//                       { return std::get<1>(lhs) < std::get<1>(rhs); });

//             int cur_armor_index_left = std::get<0>(dists[0]);

//             // 让armor_measure的yaw取值从(-PI,PI)延伸到跟整车状态的yaw一个范围(-unlitmit,+unlimit)
//             int vehicle_yaw_region = floor((armor_serial_base[cur_armor_index_left][3] + PI / 2) / 2 / PI);
//             armor_measure[3] += 2 * PI * vehicle_yaw_region;
//             armor_measure_sec[3] += 2 * PI * vehicle_yaw_region;

//             return makeMeasureVector(armor_measure, armor_measure_sec, cur_armor_index_left);
//         }
//     }

//     Eigen::Vector4d VehicleTracking::PoseToMeasurement(const Sophus::SE3 &pose)
//     {
//         Eigen::Vector4d measurement;
//         double x, y, z, yaw;
//         x = pose.translation().x();
//         y = pose.translation().y();
//         z = pose.translation().z();
//         Eigen::Vector3d eular_angles;
//         // eular_angles = pose.so3().matrix().eulerAngles(2, 0, 1);
//         // eular_angles = pose.unit_quaternion().toRotationMatrix().eulerAngles(2, 1, 0);
//         //   eular_angles = pose.so3().log();
//         Eigen::Matrix3d R = pose.so3().matrix();
//         // double pitch_ = asin(-R(2, 0));
//         // double roll_ = atan2(R(2, 1), R(2, 2));
//         // double yaw_ = atan2(R(1, 0) / cos(pitch_), R(0, 0) / cos(pitch_));
//         Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
//         Eigen::Matrix3d R_new = svd.matrixU() * svd.matrixV().transpose();
//         eular_angles = R_new.eulerAngles(2, 1, 0);
//         // eular_angles[0] = eular_angles[0] > PI / 2 ? eular_angles[0] - PI : eular_angles[0];
//         yaw = eular_angles[0];
//         // yaw = yaw_;
//         measurement << x, y, z, yaw;
//         DLOG(WARNING) << "yyyyyyyyyyyyyyyyyyyyyy" << eular_angles[0] << std::endl;
//         DLOG(WARNING) << "ppp" << eular_angles[1] << std::endl;
//         DLOG(WARNING) << "rrrrrrrrrrrrrrrrrrrrrr" << eular_angles[2] << std::endl;
//         return measurement;
//     }

//     void VehicleTracking::setUpdateTime(const double &delta_t)
//     {
//         if (fabs(delta_t) < 1) // 防止时间差为0
//         {
//             update_time = 1.0 / 1000.0;
//         }
//         else
//         {
//             update_time = delta_t / 1000.0;
//         }
//     }

//     Eigen::Matrix<double, 16, 1> VehicleTracking::makeMeasureVector(Eigen::Vector4d &armor_measure, int &cur_armor_index)
//     {
//         Eigen::Matrix<double, 9, 1> state_vector = extended_kalman_filter->post_state;
//         double r = state_vector[8]; // < 0.2 ? 0.2 : (state_vector[8] > 0.4 ? 0.4 : state_vector[8]);
//         double x_c = armor_measure[0] + r * sin(armor_measure[3]);
//         double y_c = armor_measure[1] + r * cos(armor_measure[3]);
//         double z_c = armor_measure[2];

//         // r = 0.2;
//         Eigen::Matrix<double, 16, 1> whole_car_measure;
//         Eigen::Vector4d armor_a, armor_b, armor_c, armor_d;
//         armor_a << armor_measure[0], armor_measure[1], armor_measure[2], armor_measure[3];
//         armor_b << x_c - r * sin(armor_measure[3] - PI / 2), y_c - r * cos(armor_measure[3] - PI / 2), z_c, armor_measure[3] - PI / 2;
//         armor_c << x_c - r * sin(armor_measure[3] - PI), y_c - r * cos(armor_measure[3] - PI), z_c, armor_measure[3] - PI;
//         armor_d << x_c - r * sin(armor_measure[3] + PI / 2), y_c - r * cos(armor_measure[3] + PI / 2), z_c, armor_measure[3] + PI / 2;

//         DLOG(ERROR) << "aaaaaaaaaaa" << armor_a << std::endl;
//         DLOG(ERROR) << "bbbbbbbbbbb" << armor_b << std::endl;
//         DLOG(ERROR) << "ccccccccccc" << armor_c << std::endl;
//         DLOG(ERROR) << "ddddddddddd" << armor_d << std::endl;

//         if (cur_armor_index == 0)
//         {
//             whole_car_measure << armor_a, armor_b, armor_c, armor_d;
//         }
//         else if (cur_armor_index == 1)
//         {
//             whole_car_measure << armor_d, armor_a, armor_b, armor_c;
//         }
//         else if (cur_armor_index == 2)
//         {
//             whole_car_measure << armor_c, armor_d, armor_a, armor_b;
//         }
//         else
//         {
//             whole_car_measure << armor_b, armor_c, armor_d, armor_a;
//         }

//         // DLOG(ERROR) << "WHOLE CAR MEASURE" << whole_car_measure << std::endl;
//         return whole_car_measure;
//     }

//     Eigen::Matrix<double, 16, 1> VehicleTracking::makeMeasureVector(Eigen::Vector4d &armor_measure, Eigen::Vector4d &armor_measure_sec, int &cur_left_armor_index)
//     {
//         // Eigen::Matrix<double, 9, 1> state_vector = extended_kalman_filter->post_state;
//         double r = (armor_measure[0] - armor_measure_sec[0]) / (sin(armor_measure_sec[3]) - sin(armor_measure[3])) > 0.25 ? 0.25 : (armor_measure[0] - armor_measure_sec[0]) / (sin(armor_measure_sec[3]) - sin(armor_measure[3]));
//         double x_c = armor_measure[0] + r * sin(armor_measure[3]);
//         double y_c = armor_measure[1] + r * cos(armor_measure[3]);
//         double z_c = armor_measure[2];

//         Eigen::Matrix<double, 16, 1> whole_car_measure;
//         Eigen::Vector4d armor_a, armor_b, armor_c, armor_d;
//         armor_a << armor_measure[0], armor_measure[1], armor_measure[2], armor_measure[3];
//         armor_b << armor_measure_sec[0], armor_measure_sec[1], armor_measure_sec[2], armor_measure_sec[3];
//         armor_c << x_c - r * sin(armor_measure[3] - PI), y_c - r * cos(armor_measure[3] - PI), z_c, armor_measure[3] - PI;
//         armor_d << x_c - r * sin(armor_measure_sec[3] - PI), y_c - r * cos(armor_measure_sec[3] - PI), z_c, armor_measure_sec[3] - PI;

//         if (cur_left_armor_index == 0)
//         {
//             whole_car_measure << armor_a, armor_b, armor_c, armor_d;
//             return whole_car_measure;
//         }
//         else if (cur_left_armor_index == 1)
//         {
//             whole_car_measure << armor_d, armor_a, armor_b, armor_c;
//             return whole_car_measure;
//         }
//         else if (cur_left_armor_index == 2)
//         {
//             whole_car_measure << armor_c, armor_d, armor_a, armor_b;
//             return whole_car_measure;
//         }
//         else
//         {
//             whole_car_measure << armor_b, armor_c, armor_d, armor_a;
//             return whole_car_measure;
//         }
//     }

//     // 由于update_time一直在变,状态转移矩阵需要每次重新传进ekf
//     void VehicleTracking::setTransitionMatrix()
//     {
//         extended_kalman_filter->post_state[8] = (extended_kalman_filter->post_state[8] < 0.2) ? 0.2 : (extended_kalman_filter->post_state[8] > 0.4 ? 0.4 : extended_kalman_filter->post_state[8]);
//         //   f
//         auto f = [this](const Eigen::VectorXd &x)
//         {
//             Eigen::VectorXd x_new = x;
//             x_new(0) += x(4) * update_time;
//             x_new(1) += x(5) * update_time;
//             x_new(2) += x(6) * update_time;
//             x_new(3) += x(7) * update_time;
//             return x_new;
//         };

//         // f_jet
//         auto j_f = [this](const Eigen::VectorXd &)
//         {
//             Eigen::MatrixXd f(9, 9);
//             f << 1, 0, 0, 0, update_time, 0, 0, 0, 0,
//                 0, 1, 0, 0, 0, update_time, 0, 0, 0,
//                 0, 0, 1, 0, 0, 0, update_time, 0, 0,
//                 0, 0, 0, 1, 0, 0, 0, update_time, 0,
//                 0, 0, 0, 0, 1, 0, 0, 0, 0,
//                 0, 0, 0, 0, 0, 1, 0, 0, 0,
//                 0, 0, 0, 0, 0, 0, 1, 0, 0,
//                 0, 0, 0, 0, 0, 0, 0, 1, 0,
//                 0, 0, 0, 0, 0, 0, 0, 0, 1;
//             return f;
//         };

//         extended_kalman_filter->transition_func = f;
//         extended_kalman_filter->transition_func_jet = j_f;
//     }

//     void VehicleTracking::setObservationMatrix()
//     {
//         // h
//         auto h = [](const Eigen::VectorXd &x)
//         {
//             Eigen::VectorXd z(16);
//             double xc = x(0), yc = x(1), yaw = x(3), r = x(8);
//             z(0) = xc - r * sin(yaw);           // x_1
//             z(1) = yc - r * cos(yaw);           // y_1
//             z(2) = x(2);                        // z_1
//             z(3) = x(3);                        // yaw_1
//             z(4) = xc - r * sin(yaw - PI / 2);  // x_2
//             z(5) = yc - r * cos(yaw - PI / 2);  // y_2
//             z(6) = x(2);                        // z_2
//             z(7) = x(3) - PI / 2;               // yaw_2
//             z(8) = xc - r * sin(yaw - PI);      // x_3
//             z(9) = yc - r * cos(yaw - PI);      // y_3
//             z(10) = x(2);                       // z_3
//             z(11) = x(3) - PI;                  // yaw_3
//             z(12) = xc - r * sin(yaw + PI / 2); // x_4
//             z(13) = yc - r * cos(yaw + PI / 2); // y_4
//             z(14) = x(2);                       // z_4
//             z(15) = x(3) + PI / 2;              // yaw_4
//             return z;
//         };

//         // h_jet
//         auto j_h = [](const Eigen::VectorXd &x)
//         {
//             Eigen::MatrixXd h(16, 9);
//             double yaw = x(3), r = x(8);
//             //  xc   yc   zc   yaw  vxc  vyc  vzc  vyaw  r
//             h << 1, 0, 0, -r * cos(yaw), 0, 0, 0, 0, -sin(yaw),
//                 0, 1, 0, r * sin(yaw), 0, 0, 0, 0, -cos(yaw),
//                 0, 0, 1, 0, 0, 0, 0, 0, 0,
//                 0, 0, 0, 1, 0, 0, 0, 0, 0,
//                 1, 0, 0, -r * cos(yaw - PI / 2), 0, 0, 0, 0, -sin(yaw - PI / 2),
//                 0, 1, 0, r * sin(yaw - PI / 2), 0, 0, 0, 0, -cos(yaw - PI / 2),
//                 0, 0, 1, 0, 0, 0, 0, 0, 0,
//                 0, 0, 0, 1, 0, 0, 0, 0, 0,
//                 1, 0, 0, -r * cos(yaw - PI), 0, 0, 0, 0, -sin(yaw - PI),
//                 0, 1, 0, r * sin(yaw - PI), 0, 0, 0, 0, -cos(yaw - PI),
//                 0, 0, 1, 0, 0, 0, 0, 0, 0,
//                 0, 0, 0, 1, 0, 0, 0, 0, 0,
//                 1, 0, 0, -r * cos(yaw + PI / 2), 0, 0, 0, 0, -sin(yaw + PI / 2),
//                 0, 1, 0, r * sin(yaw + PI / 2), 0, 0, 0, 0, -cos(yaw + PI / 2),
//                 0, 0, 1, 0, 0, 0, 0, 0, 0,
//                 0, 0, 0, 1, 0, 0, 0, 0, 0;
//             return h;
//         };

//         extended_kalman_filter->observe_func = h;
//         extended_kalman_filter->observe_func_jet = j_h;
//     }

//     void VehicleTracking::setQandRMatrix()
//     {
//         Eigen::DiagonalMatrix<double, 9> q;
//         q.diagonal() << 0.1, 0.1, 0.1, 0.2, 8, 8, 0.01, 8, 0.001;
//         extended_kalman_filter->Q = q;

//         Eigen::DiagonalMatrix<double, 16> r;
//         r.diagonal() << 0.01, 0.01, 0.01, 0.2, 0.01, 0.01, 0.01, 0.2, 0.01, 0.01, 0.01, 0.2, 0.01, 0.01, 0.01, 0.2;
//         extended_kalman_filter->R = r;
//     }

//     // void VehicleTracking::vizDebug()
//     // {

//     //     window.setWindowSize(cv::Size(1280, 1024));

//     //     coordinate_system.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
//     //     coordinate_system.setRenderingProperty(cv::viz::OPACITY, 1.0);
//     //     window.showWidget("world", coordinate_system);

//     //     window.spinOnce(0.1, false);
//     // }
// }

#include "VehicleTracking.h"
namespace ly
{

#define PI 3.1415926

    VehicleTracking::VehicleTracking()
    {
        extended_kalman_filter = new ExtendedKalman<double, 9, 4>();
        is_kalman_init = false;

        // 观测矩阵一直保持不变
        setObservationMatrix();
    }

    VehicleTracking::~VehicleTracking()
    {
        delete extended_kalman_filter;
    }

    //    NOT_GET_TARGET = 0,       // 未获得目标
    //    CONTINOUS_GET_TARGET = 1, // 连续获得目标
    //    LOST_BUMP = 2,            // 缓冲阶段
    //    DETECT_BUMP = 3           // 进入连续识别状态的缓冲
    Eigen::Vector3d VehicleTracking::predictVehicleState(const Sophus::SE3 &armor_pose, const Sophus::SE3 &armor_pose_sec, bool is_get_second_armor, int &detect_mode, float shoot_time, float frame_delta_t, float yaw_gimbal)
    {
        // std::cout << "tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt" << frame_delta_t << std::endl;
        armor_switch = false;

        Eigen::Vector3d predict_point;
        Eigen::Vector4d measure, measure_sec, tracked_measure;

        measure = PoseToMeasurement(armor_pose);

        if (is_get_second_armor)
        {
            measure_sec = PoseToMeasurement(armor_pose_sec);
        }
        yaw_send = measure[3];
        // measure[3] = measure[3] + yaw_gimbal_norm > PI / 2 ? measure[3] - PI : (measure[3] + yaw_gimbal_norm < -PI / 2 ? measure[3] + PI : measure[3]);
        // measure_sec[3] = measure_sec[3] + yaw_gimbal_norm > PI / 2 ? measure_sec[3] - PI : (measure_sec[3] + yaw_gimbal_norm < -PI / 2 ? measure_sec[3] + PI : measure_sec[3]);

        if (!is_kalman_init)
        {
            is_kalman_init = true;
            rebootKalman(is_get_second_armor, measure, measure_sec);
            setUpdateTime(frame_delta_t);
            setQandRMatrix();
        }
        else
        {
            setUpdateTime(frame_delta_t);
            setTransitionMatrix();
            setQandRMatrix();
            Eigen::Matrix<double, 9, 1> prediction = extended_kalman_filter->predict();
            vehicle_state = prediction;

            if (detect_mode != 2) // 不是LOST BUMP,whole_car_state需要测量更新
            {
                double min_position_diff = 0;
                if (is_get_second_armor)
                {
                    Eigen::Vector3d predicted_position = getArmorPositionFromState(prediction);
                    Eigen::Vector3d position_vec_1(measure[0], measure[1], measure[2]);
                    Eigen::Vector3d position_vec_2(measure_sec[0], measure_sec[1], measure_sec[2]);
                    double position_diff_1 = (position_vec_1 - predicted_position).norm();
                    double position_diff_2 = (position_vec_2 - predicted_position).norm();

                    tracked_measure = position_diff_1 < position_diff_2 ? measure : measure_sec;
                    min_position_diff = position_diff_1 < position_diff_2 ? position_diff_1 : position_diff_2;
                }
                else
                {
                    tracked_measure = measure;
                    Eigen::Vector3d predicted_position = getArmorPositionFromState(prediction);
                    Eigen::Vector3d position_vec_1(measure[0], measure[1], measure[2]);
                    min_position_diff = (position_vec_1 - predicted_position).norm();
                }
                if (min_position_diff < 0.2) // matched
                {
                    vehicle_state = extended_kalman_filter->update(tracked_measure);
                }
                else
                {
                    handleArmorJump(tracked_measure);
                }
            }
            if (vehicle_state(8) < 0.2)
            {
                vehicle_state(8) = 0.2;
                extended_kalman_filter->post_state = vehicle_state;
            }
        }

        DLOG(WARNING) << "WHOLE CAR STATE" << std::endl
                      << vehicle_state << std::endl;

        predict_point = getPredictPoint(vehicle_state, shoot_time, yaw_gimbal);

        DLOG(WARNING) << "PREDICT POINT" << std::endl
                      << predict_point << std::endl;

        return predict_point;
    }

    void VehicleTracking::getVehicleState(Eigen::Vector4d &measure)
    {
        last_yaw = 0;
        double x_a = measure[0];
        double y_a = measure[1];
        double z_a = measure[2];
        double yaw = measure[3];

        double r = 0.2;
        double x_c = x_a + r * sin(yaw);
        double y_c = y_a + r * cos(yaw);
        double z_c = z_a;
        last_z = z_c, last_r = r;

        vehicle_state << x_c, 0, y_c, 0, z_c, 0, yaw, 0, r;
    }

    Eigen::Vector3d VehicleTracking::getArmorPositionFromState(Eigen::Matrix<double, 9, 1> x)
    {
        double xc = x(0), yc = x(2), zc = x(4);
        double yaw = x(6), r = x(8);
        double xa = xc - r * sin(yaw);
        double ya = yc - r * cos(yaw);
        return Eigen::Vector3d(xa, ya, zc);
    }

    void VehicleTracking::resetTracker()
    {
        is_kalman_init = false;
    }

    void VehicleTracking::rebootKalman(bool is_get_sec, Eigen::Vector4d armor_1, Eigen::Vector4d armor_2)
    {
        if (is_get_sec)
        {
            if (armor_1[0] * armor_1[0] + armor_1[1] * armor_1[1] >= armor_2[0] * armor_2[0] + armor_2[1] * armor_2[1])
            {
                getVehicleState(armor_1);
            }
            else
            {
                getVehicleState(armor_2);
            }
        }
        else
        {
            getVehicleState(armor_1);
        }
        extended_kalman_filter->post_state = vehicle_state;
        extended_kalman_filter->P_post = Eigen::Matrix<double, 9, 9>::Identity();
    }

    void VehicleTracking::handleArmorJump(Eigen::Vector4d &measure)
    {
        double last_yaw = vehicle_state(6);
        double yaw = measure[3];

        if (abs(measure[3] - last_yaw) > 0.2)
        {
            last_z = vehicle_state(4);
            vehicle_state(4) = measure[2];
            vehicle_state(6) = measure[3];
            std::swap(vehicle_state(8), last_r);

            armor_switch = true;
            DLOG(WARNING) << "SWITCH ARMOR" << std::endl;
        }

        Eigen::Vector3d current_position(measure[0], measure[1], measure[2]);
        Eigen::Vector3d infer_position = getArmorPositionFromState(vehicle_state);

        if ((current_position - infer_position).norm() > 0.2)
        {
            double r = vehicle_state(8);
            vehicle_state(0) = measure[0] + r * sin(yaw);
            vehicle_state(2) = measure[1] + r * cos(yaw);
            vehicle_state(1) = 0;
            vehicle_state(3) = 0;
            DLOG(ERROR) << "STATE WENT WRONG" << std::endl;
        }

        extended_kalman_filter->post_state = vehicle_state;
    }

    Eigen::Vector3d VehicleTracking::getPredictPoint(const Eigen::Matrix<double, 9, 1> vehicle_state, float &shoot_t, float yaw_gimbal)
    {
        float shoot_t_ = shoot_t + ShootParam::shoot_delay;
        vector<Eigen::Vector4d> armor_serial_with_angle;
        armor_serial.clear();
        bool use_1 = true;
        double yaw = vehicle_state(6);
        double r1 = vehicle_state(8), r2 = last_r;
        double x_c = vehicle_state(0), y_c = vehicle_state(2), z_c = vehicle_state(4), z_2 = last_z;
        double v_x = vehicle_state(1), v_y = vehicle_state(3), v_yaw = vehicle_state(7);
        double x_c_pre = x_c + v_x * shoot_t_, y_c_pre = y_c + v_y * shoot_t_, yaw_pre = yaw + v_yaw * shoot_t_;

        // 用vehicle_state做更新
        for (int i = 0; i < 4; i++)
        {
            Eigen::Vector3d p;
            double temp_yaw = yaw_pre + i * PI / 2;
            double r = use_1 ? r1 : r2;

            double x = x_c_pre - r * sin(temp_yaw);
            double y = y_c_pre - r * cos(temp_yaw);
            double z = use_1 ? z_c : z_2;

            p << x, y, z;
            armor_serial.push_back(p);
            armor_serial_with_angle.push_back({x, y, z, temp_yaw});
            use_1 = !use_1;
        }

        speed_vector.clear();
        speed_vector.push_back(vehicle_state[1]);
        speed_vector.push_back(vehicle_state[3]);
        speed_vector.push_back(vehicle_state[7]);
        speed_vector.push_back(shoot_t);

        // 对armor_serial进行排序，击打最优装甲板
        std::vector<Eigen::Vector3d> armor_serial_temp = armor_serial;

        // // 生成angle排序索引
        // std::vector<int> angle_indices(angle_vec.size());
        // std::iota(angle_indices.begin(), angle_indices.end(), 0);
        // std::sort(angle_indices.begin(), angle_indices.end(), [&angle_vec, &yaw_gimbal](int a, int b)
        //           {
        //     if (angle_vec[a]!=angle_vec[b])
        //         return angle_vec[a]+yaw_gimbal<angle_vec[a]+yaw_gimbal; });

        // // 按照排序索引对armor_serial_temp进行排序
        // std::vector<Eigen::Vector3d> sorted_armor_serial(armor_serial_temp.size());
        // std::transform(angle_indices.begin(), angle_indices.end(), sorted_armor_serial.begin(), [&](int i)
        //                { return armor_serial_temp[i]; });
        // armor_serial_temp.assign(sorted_armor_serial.begin(), sorted_armor_serial.end());
        // std::sort(armor_serial_temp.begin(), armor_serial_temp.end(), [](const Eigen::Vector3d &a, const Eigen::Vector3d &b)
        //          { return std::abs(a.norm()) < std::abs(b.norm()); });
        int region = floor(yaw_gimbal / (2 * PI));
        float yaw_gimbal_norm = yaw_gimbal - region * 2 * PI;
        std::sort(armor_serial_with_angle.begin(), armor_serial_with_angle.end(), [&yaw_gimbal_norm](const Eigen::Vector4d &a, const Eigen::Vector4d &b)
                  { return std::abs(a[3] + yaw_gimbal_norm) < std::abs(b[3] + yaw_gimbal_norm); });
        Eigen::Vector3d armor_to_hit{armor_serial_with_angle[0][0], armor_serial_with_angle[0][1], armor_serial_with_angle[0][2]};
        return armor_to_hit;
    }

    // double shortest_angle_distance(double from, double to)
    // {
    //     double result = normalize_angle(to) - normalize_angle(from);
    //     if (result > PI)
    //         result -= 2 * PI;
    //     else if (result < -PI)
    //         result += 2 * PI;
    //     return result;
    // }

    Eigen::Vector4d VehicleTracking::PoseToMeasurement(const Sophus::SE3 &pose)
    {
        Eigen::Vector4d measurement;
        double x, y, z, yaw;
        x = pose.translation().x();
        y = pose.translation().y();
        z = pose.translation().z();
        Eigen::Vector3d eular_angles;
        // eular_angles = pose.so3().matrix().eulerAngles(2, 0, 1);
        // eular_angles = pose.unit_quaternion().toRotationMatrix().eulerAngles(2, 1, 0);
        //   eular_angles = pose.so3().log();
        Eigen::Matrix3d R = pose.so3().matrix().inverse();
        // double pitch_ = asin(-R(2, 0));
        // double roll_ = atan2(R(2, 1), R(2, 2));
        // double yaw_ = atan2(R(1, 0) / cos(pitch_), R(0, 0) / cos(pitch_));
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_new = svd.matrixU() * svd.matrixV().transpose();
        eular_angles = R_new.eulerAngles(2, 1, 0);
        // eular_angles[0] = eular_angles[0] > PI / 2 ? eular_angles[0] - PI : eular_angles[0];
        yaw = eular_angles[0];
        // yaw = last_yaw + shortest_angle_distance(last_yaw, yaw);
        last_yaw = yaw;
        // yaw = yaw_;
        measurement << x, y, z, yaw;

        return measurement;
    }

    void VehicleTracking::setUpdateTime(const double &delta_t)
    {
        if (fabs(delta_t) < 5e-3) // 防止时间差为0
        {
            update_time = 5.0 / 1000.0;
        }
        else
        {
            update_time = delta_t / 1000.0;
        }
    }

    // 由于update_time一直在变,状态转移矩阵需要每次重新传进ekf
    void VehicleTracking::setTransitionMatrix()
    {
        extended_kalman_filter->post_state[8] = (extended_kalman_filter->post_state[8] < 0.2) ? 0.2 : (extended_kalman_filter->post_state[8] > 0.4 ? 0.4 : extended_kalman_filter->post_state[8]);
        //   f
        auto f = [this](const Eigen::VectorXd &x)
        {
            Eigen::VectorXd x_new = x;
            x_new(0) += x(1) * update_time;
            x_new(2) += x(3) * update_time;
            x_new(4) += x(5) * update_time;
            x_new(6) += x(7) * update_time;
            return x_new;
        };

        // f_jet
        auto j_f = [this](const Eigen::VectorXd &)
        {
            Eigen::MatrixXd f(9, 9);
            f << 1, update_time, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, update_time, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, update_time, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, update_time, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 1;
            return f;
        };

        extended_kalman_filter->transition_func = f;
        extended_kalman_filter->transition_func_jet = j_f;
    }

    void VehicleTracking::setObservationMatrix()
    {
        // h
        auto h = [](const Eigen::VectorXd &x)
        {
            Eigen::VectorXd z(4);
            double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
            z(0) = xc - r * sin(yaw); // x
            z(1) = yc - r * cos(yaw); // y
            z(2) = x(4);              // z
            z(3) = x(6);              // yaw
            return z;
        };

        // h_jet
        auto j_h = [](const Eigen::VectorXd &x)
        {
            Eigen::MatrixXd h(4, 9);
            double yaw = x(6), r = x(8);
            //  xc   v_x   y_c   v_y  z_c   v_z   yaw  v_yaw  r
            h << 1, 0, 0, 0, 0, 0, -r * cos(yaw), 0, -sin(yaw),
                0, 0, 1, 0, 0, 0, r * sin(yaw), 0, -cos(yaw),
                0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0;
            return h;
        };

        extended_kalman_filter->observe_func = h;
        extended_kalman_filter->observe_func_jet = j_h;
    }

    void VehicleTracking::setQandRMatrix()
    {
        Eigen::Matrix<double, 2, 1> process_noise_vec;
        process_noise_vec << 0.5 * update_time * update_time, update_time;
        Eigen::Matrix<double, 6, 3> process_noise_matrix_xyz = Eigen::Matrix<double, 6, 3>::Zero();
        for (int i = 0; i < 3; i++)
        {
            process_noise_matrix_xyz.block<2, 1>(2 * i, i) = process_noise_vec;
        }
        Eigen::Matrix3d process_noise_xyz = Eigen::Matrix3d::Identity();
        process_noise_xyz.diagonal() << FilterParams::process_noise_x_c, FilterParams::process_noise_y_c, FilterParams::process_noise_z_c;
        Eigen::Matrix<double, 6, 6> q_xyz = process_noise_matrix_xyz * process_noise_xyz * process_noise_matrix_xyz.transpose();
        Eigen::Matrix<double, 2, 1> process_noise_matrix_yaw = process_noise_vec;
        Eigen::Matrix<double, 2, 2> q_yaw = process_noise_matrix_yaw * FilterParams::process_noise_yaw * process_noise_matrix_yaw.transpose();
        Eigen::Matrix<double, 9, 9> q = Eigen::Matrix<double, 9, 9>::Identity();
        q.block<6, 6>(0, 0) = q_xyz;
        q.block<2, 2>(6, 6) = q_yaw;
        q(8, 8) = FilterParams::process_noise_r;
        extended_kalman_filter->Q = q;

        Eigen::DiagonalMatrix<double, 4> r;
        r.diagonal() << 0.005, 0.005, 0.005, 0.05;
        extended_kalman_filter->R = r;
    }

}