#include "ROI_Accelerator.h"
namespace ly
{   
    // 初始全图
    ROIAccelerator::ROIAccelerator()
    {
        roi = cv::Rect(0, 0, 1280, 1024);
        roi_shut_down = true;
    }

    ROIAccelerator::~ROIAccelerator()
    {
    }

    // 生成ROI
    void ROIAccelerator::ROI_create(const std::vector<cv::Point2f> &corners)
    {
        // 根据优先级判断
        cv::Point2f center = cv::Point2f(0, 0);
        for (int i = 0; i < corners.size(); i++)
        {
            center += corners[i];
        }
        center /= 4; // 求得中心

        cv::Point2f point_diff_width = (corners[1] + corners[2] - corners[0] - corners[3]) / 2;
        cv::Point2f point_diff_height = (corners[2] + corners[3] - corners[1] - corners[0]) / 2;

        // 装甲板的宽高
        float WIDTH = sqrt(point_diff_width.x * point_diff_width.x + point_diff_width.y * point_diff_width.y);
        float HEIGHT = sqrt(point_diff_height.x * point_diff_height.x + point_diff_height.y * point_diff_height.y);

        float width_ratio = 10.f;       // 这两个比例，相当于两个超参
        float height_ratio = 8.0f;

        // 最低高度像素
        if (HEIGHT < 50) // 设定最低高度区域
        {
            HEIGHT = 50;
        }

        // 暂时分配固定大小的ROI
        // 按照一定比例扩大
        cv::Point tl_point = cv::Point((int)(center.x - WIDTH * width_ratio), (int)(center.y - HEIGHT * height_ratio));
        cv::Point br_point = cv::Point((int)(center.x + WIDTH * width_ratio), (int)(center.y + HEIGHT * height_ratio));

        // 限制在全图范围内
        makePointInRange(tl_point);
        makePointInRange(br_point);

        // DLOG(WARNING) << tl_point << "----" <<br_point << std::endl;

        roi = cv::Rect(tl_point, br_point);
    }

    // 重新分配POI
    void ROIAccelerator::ROI_destroy()
    {
        this->roi = cv::Rect(0, 0, 1280, 1024); // 全图搜索
    }

    // 获取roi的偏移
    const cv::Point ROIAccelerator::getRoiOffset()
    {
        return roi.tl();
    }


    void ROIAccelerator::makePointInRange(cv::Point &point)
    {
        point.x = point.x > 1280 ? 1280 : (point.x < 0 ? 0 : point.x);
        point.y = point.y > 1024 ? 1024 : (point.y < 0 ? 0 : point.y);
    }
}