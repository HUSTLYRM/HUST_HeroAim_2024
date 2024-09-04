#ifndef VIDEO_SAVER
#define VIDEO_SAVER

#include "opencv2/opencv.hpp"
#include "VideoCapture.h"
#include "Params.h"

using namespace cv;

namespace ly {
    class VideoSaver{
    public: 
        VideoSaver();
        ~VideoSaver();
        void SaveVideo(Image** frame_p);
    private:
        int id;
        float gamma;
        unordered_map<int, float> gamma_table;
        VideoWriter writer;
        VideoWriter writer_visual;
    };
}
#endif