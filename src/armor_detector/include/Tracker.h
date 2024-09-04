//
// Created by zhiyu on 2021/8/30.
//

#ifndef AUTOAIM_TRACKER_H
#define AUTOAIM_TRACKER_H

#include "opencv2/opencv.hpp"

#include "Log.h"

#define DEBUG

using namespace cv;

namespace ly{
    enum State{
        OCCLUDE,
        LOST,
        FOUND
    };

    struct Image{
        Mat frame;
        Mat image_sepctrum;
        Mat filter_output;

        int cols;
        int rows;

        int optimal_dft_rows;
        int optimal_dft_cols;
    };

    struct Roi_{
        Rect2i roi;
        Point2i roi_center;
    };

    class Tracker {
    public:
        explicit Tracker();
        ~Tracker() = default;

        Image curr_image;
        Image prev_image;

        Roi_ curr_roi;
        Roi_ prev_roi;

        void initTracker(const Mat&, Point2i, Point2i);
        void initTracker(const Mat&, Rect);

        void track(const Mat& frame);

        bool isInit() const;

    private:
        State _state;
        bool _init;
        double _learning_rate;
        Size _image_size;
        Mat _filter;
        Mat _hanning_window;
        bool _eps;
        int _PSR_mask;
        double _PSR_ratio[2];

        void calcDFT(Image& input_image, bool preprocess);
        Mat calcDFT(const Mat& input_image, bool preprocess);

        void setRoi(const Rect&);

        void initFilter();

        void maskDesiredG(Mat& output, int u_x, int u_y, double sigma = 2, bool norm_energy = true);
        Mat createEps(const Mat& input_, double std = 1e-5);
        void AffineTransform(const Mat &input_image, const Mat &input_image2, Mat &aff_img, Mat &aff_img2);
        void dftDiv(const Mat &dft_a, const Mat &dft_b, Mat &output_dft);

        Point2i performTrack();
        float calcPSR(const Mat& correlation_mat);
        void update(Point2i new_location);
        void updateFilter();
        void updateRoi(Point2i new_center, bool scale_rot);
    };
}


#endif //AUTOAIM_TRACKER_H
