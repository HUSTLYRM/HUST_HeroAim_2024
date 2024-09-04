#ifndef NUMBER_CLASSIFIER_H
#define NUMBER_CLASSIFIER_H
#include "Config.h"
#include "opencv2/opencv.hpp"
#include "opencv2/ml/ml.hpp"
#include <string>
#define ROI_MEAN_THRESH 185
#define ROI_THRESH_MIN 20
namespace ly
{
    class NumberClassifier
    {
    private:
        int mode_;

        void calcGammaTable(float gamma);
        uchar gamma_table[256];
        cv::Mat lut_table;

        void AutoGamma(const cv::Mat &img, cv::Mat &out);
        cv::Mat AutoGammaCorrect(const cv::Mat image);
        cv::Mat pixel(const cv::Mat image, double gamma, double mu, double sigma, int flag);
        bool loadModel(const std::string &model_path);

        bool is_model_set;
        cv::Ptr<cv::ml::SVM> number_svm_model;

        bool is_get_roi;
        cv::Mat number_roi;
        cv::Mat class_;

        std::pair<bool, vector<cv::Mat>> affineMultiNumber(Mat &frame, const std::vector<cv::Point2f> &corners);
        bool affineNumber(Mat &frame, const std::vector<cv::Point2f> &corners);
        float getDistance(const cv::Point2f &point_1, const cv::Point2f &point_2);
        void saveImage();
        void showNumber();

        std::string armor_classify_type[9] = {"Undefined", "Hero", "Engineer", "Infantry", "Infantry", "Infantry", "Sentry", "Outpost", "Base"};

        // debug用
        cv::Mat armor_to_show;
        cv::Mat save;

        cv::HOGDescriptor *hog_;
        void initHog();

        // mlp+otsu
        cv::dnn::Net net_;
        std::vector<std::string> class_names_;

        // resnet
        void softmax(Mat &mat);

    public:
        NumberClassifier(int mode);
        ~NumberClassifier();
        std::pair<int, double> predict(Mat &frame, const std::vector<cv::Point2f> &corners, int find_mode);
        void showNumberStr(cv::Mat &drawing, int id, double conf, cv::Rect rect);

        int numbers_ = 0;
    };
}

#endif