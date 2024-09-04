#ifndef _VERTEX_AB_HPP_
#define _VERTEX_AB_HPP_
#include <g2o/core/base_vertex.h>

namespace ly
{
    class VertexAB : public g2o::BaseVertex<2, Eigen::Vector2d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 字节对齐
            virtual void
            setToOriginImpl() // 重置，设定被优化变量的原始值
        {
            _estimate << 0, 0;
        }

        virtual void oplusImpl(const double *update) // 更新
        {
            _estimate += Eigen::Vector2d{update[0], update[1]}; // update
        }
        // 存盘和读盘：留空
        virtual bool read(std::istream &in) {}
        virtual bool write(std::ostream &out) const {}
    };
}
#endif