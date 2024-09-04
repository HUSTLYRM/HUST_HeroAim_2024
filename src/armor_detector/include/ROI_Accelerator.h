#ifndef _ROI_ACCELERATOR_H
#define _ROI_ACCELERATOR_H
#include "opencv2/opencv.hpp"
#include "Config.h"
namespace ly
{
    class ROIAccelerator
    {

    public:
        ROIAccelerator();
        ~ROIAccelerator();
        void ROI_create(const std::vector<cv::Point2f> &corners);
        void ROI_destroy();
        const cv::Point getRoiOffset();
        const cv::Rect &getROI()
        {
            return this->roi;
        }
        void drawROI(cv::Mat &frame)
        {
            
            if (GlobalParam::DEBUG_MODE)
            {
                // DLOG(INFO) << "drawing ROI" << std::endl;
                cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2, 8);
            }
        }

    private:
        bool roi_shut_down;

        cv::Rect roi;

        void makePointInRange(cv::Point &point);
    };

}
#endif