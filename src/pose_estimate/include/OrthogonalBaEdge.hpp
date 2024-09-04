#ifndef _ORTHOGONAL_BA_EDGE_HPP_
#define _ORTHOGONAL_BA_EDGE_HPP_
#include <g2o/core/base_binary_edge.h>
#include "VertexAB.hpp"
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "Params.h"

namespace ly
{
    class OrthogonalEdge : public g2o::BaseBinaryEdge<8, Eigen::Matrix<double, 4, 2>, VertexAB, g2o::VertexSE3Expmap>
    {
    public:
        OrthogonalEdge()
        {
            fx = ly::CameraParam::fx;
            fy = ly::CameraParam::fy;
            u = ly::CameraParam::u0;
            v = ly::CameraParam::v0;
            K << fx, 0, u, 0, fy, v, 0, 0, 1;
        }
        virtual bool read(std::istream &in) {}
        virtual bool write(std::ostream &out) const {}
        virtual void computeError() override
        {
            const VertexAB *a_and_b = static_cast<const VertexAB *>(_vertices[0]);
            const g2o::VertexSE3Expmap *pose = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
            Eigen::Matrix<double, 4, 2> obs(_measurement);
            R = pose->estimate().rotation().matrix();
            T = pose->estimate().translation().matrix();
            Eigen::Vector3d t = Eigen::Vector3d{T[0], T[1], T[2]};
            Eigen::Vector3d reproject_point1, reproject_point2, reproject_point3, reproject_point4;
            reproject_point1 = K * (R * Eigen::Vector3d{-a_and_b->estimate().matrix()[0], -a_and_b->estimate().matrix()[1], 0} + t);
            reproject_point2 = K * (R * Eigen::Vector3d{a_and_b->estimate().matrix()[0], -a_and_b->estimate().matrix()[1], 0} + t);
            reproject_point3 = K * (R * Eigen::Vector3d{a_and_b->estimate().matrix()[0], a_and_b->estimate().matrix()[1], 0} + t);
            reproject_point4 = K * (R * Eigen::Vector3d{-a_and_b->estimate().matrix()[0], a_and_b->estimate().matrix()[1], 0} + t);

            reproject_point1 = reproject_point1 / reproject_point1[2];
            reproject_point2 = reproject_point2 / reproject_point2[2];
            reproject_point3 = reproject_point3 / reproject_point3[2];
            reproject_point4 = reproject_point4 / reproject_point4[2];

            x_diff[0] = obs(0, 0) - reproject_point1[0];
            x_diff[1] = obs(1, 0) - reproject_point2[0];
            x_diff[2] = obs(2, 0) - reproject_point3[0];
            x_diff[3] = obs(3, 0) - reproject_point4[0];

            y_diff[0] = obs(0, 1) - reproject_point1[1];
            y_diff[1] = obs(1, 1) - reproject_point2[1];
            y_diff[2] = obs(2, 1) - reproject_point3[1];
            y_diff[3] = obs(3, 1) - reproject_point4[1];

            _error(0) = x_diff[0];
            _error(1) = y_diff[0];
            _error(2) = x_diff[1];
            _error(3) = y_diff[1];
            _error(4) = x_diff[2];
            _error(5) = y_diff[2];
            _error(6) = x_diff[3];
            _error(7) = y_diff[3];

            // cout << "error:" << endl
            //      << _error << endl;
        }

        virtual void linearizeOplus() override
        {
            const VertexAB *a_and_b = static_cast<const VertexAB *>(_vertices[0]);
            const g2o::VertexSE3Expmap *pose = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
            g2o::SE3Quat Transform(pose->estimate());
            double a = a_and_b->estimate().matrix()[0];
            double b = a_and_b->estimate().matrix()[1];
            double ab_symbol[8] = {-1, 1, 1, -1, -1, -1, 1, 1};
            _jacobianOplusXi(0, 0) = 0.;
            _jacobianOplusXi(0, 1) = 0.;
            _jacobianOplusXi(1, 0) = 0.;
            _jacobianOplusXi(1, 1) = 0.;

            _jacobianOplusXj(0, 0) = 0.;
            _jacobianOplusXj(0, 1) = 0.;
            _jacobianOplusXj(0, 2) = 0.;
            _jacobianOplusXj(0, 3) = 0.;
            _jacobianOplusXj(0, 4) = 0.;
            _jacobianOplusXj(0, 5) = 0.;

            _jacobianOplusXj(1, 0) = 0.;
            _jacobianOplusXj(1, 1) = 0.;
            _jacobianOplusXj(1, 2) = 0.;
            _jacobianOplusXj(1, 3) = 0.;
            _jacobianOplusXj(1, 4) = 0.;
            _jacobianOplusXj(1, 5) = 0.;
            for (int i = 0; i < 4; i++)
            {
                Eigen::Vector3d point_3d = {a * ab_symbol[i], b * ab_symbol[i + 4], 0};
                Eigen::Vector3d project_point = Transform.map(point_3d);
                double x = project_point[0];
                double y = project_point[1];
                double z = project_point[2];
                double z_2 = z * z;
                Eigen::Matrix<double, 2, 3, Eigen::ColMajor> tmp;
                tmp(0, 0) = fx;
                tmp(0, 1) = 0;
                tmp(0, 2) = -x / z * fx;
                tmp(1, 0) = 0;
                tmp(1, 1) = fy;
                tmp(1, 2) = -y / z * fy;

                Eigen::Matrix<double, 2, 3> temp_3 = -1 / z * tmp * Transform.rotation().toRotationMatrix();
                Eigen::Matrix2d temp_m = (-1 / z * tmp * Transform.rotation().toRotationMatrix()).block<2, 2>(0, 0);

                // 关于a,b雅可比
                _jacobianOplusXi(2 * i, 0) = ab_symbol[i] * temp_m(0, 0);
                _jacobianOplusXi(2 * i, 1) = ab_symbol[i + 4] * temp_m(0, 1);
                _jacobianOplusXi(2 * i + 1, 0) = ab_symbol[i] * temp_m(1, 0);
                _jacobianOplusXi(2 * i + 1, 1) = ab_symbol[i + 4] * temp_m(1, 1);

                // 关于位姿雅可比

                _jacobianOplusXj(2 * i, 0) = x * (y / z_2) * fx;
                _jacobianOplusXj(2 * i, 1) = -(1 + (x * x) / z_2) * fx;
                _jacobianOplusXj(2 * i, 2) = y / z * fx;
                _jacobianOplusXj(2 * i, 3) = -1. / z * fx;
                _jacobianOplusXj(2 * i, 4) = 0;
                _jacobianOplusXj(2 * i, 5) = x / z_2 * fx;

                _jacobianOplusXj(2 * i + 1, 0) = (1 + y * y / z_2) * fy;
                _jacobianOplusXj(2 * i + 1, 1) = -x * y / z_2 * fy;
                _jacobianOplusXj(2 * i + 1, 2) = -x / z * fy;
                _jacobianOplusXj(2 * i + 1, 3) = 0;
                _jacobianOplusXj(2 * i + 1, 4) = -1. / z * fy;
                _jacobianOplusXj(2 * i + 1, 5) = y / z_2 * fy;
            }
            // cout << "J_xi:" << endl
            //      << _jacobianOplusXi << endl
            //      << "J_xj" << endl
            //      << _jacobianOplusXj << endl;
            // cout << "......" << endl;
        }

    private:
        double fx, fy, u, v;
        Eigen::Matrix<double, 3, 3> K;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        double x_diff[4];
        double y_diff[4];
        int x_symbol[4];
        int y_symbol[4];
    };
}
#endif