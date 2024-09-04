//
// Created by zhiyu on 2021/8/24.
//

#ifndef AUTOAIM_LIGHTBARFINDER_H
#define AUTOAIM_LIGHTBARFINDER_H

#include "opencv2/opencv.hpp"
#include <vector>

#include "Params.h"
#include "Log.h"

using namespace cv;
using namespace std;

namespace ly{
//    class LightBarBlob{
//    public:
//        RotatedRect rrect;
//    };

    typedef vector<RotatedRect> LightBarBlobs;

    class LightBarFinder {
    public:
        LightBarFinder();
        bool findLightBarBlobs(const Mat& frame, LightBarBlobs& lightBarBlobs);
        static inline bool isValidLightBarBlob(const RotatedRect&);

    private:
        static inline bool checkAspectRatio(double);
        static inline bool checkArea(double);
        static inline bool checkAngle(const RotatedRect&);

        Mat kernel1;
        Mat kernel2;
    };
}


#endif //AUTOAIM_LIGHTBARFINDER_H
