//
// Created by narrow on 2022/4/24.
//

#ifndef CLS_DEMO_CLASSIFIER_H
#define CLS_DEMO_CLASSIFIER_H

#include "opencv2/opencv.hpp"
#include "Log.h"
#include <vector>

using namespace cv;
using namespace std;

namespace ly{

    class Classifier {
    public:
        virtual int predict(const Mat& frame) = 0;
        virtual void train() = 0;
    };

    class NSVM : public Classifier {
    public:
        NSVM();
        int predict(const Mat& frame) override;
        void train() override;
        const Mat& getArmor();

    private:
        Ptr<ml::SVM> svm;
        Ptr<CLAHE> clahe;

        Mat armor;
        Mat final;
        Mat cls;
        int thresh; // unused
        uchar gamma_table[256];
    };

    class FSVM : public Classifier {
    public:
        FSVM();
        int predict(const Mat& frame) override;
        void train() override;
        const Mat& getArmor();

    private:
        Ptr<ml::SVM> svm;
        HOGDescriptor hog;

        Mat armor;
        vector<float> decs;
    };

    class CNN : public Classifier {
    public:
        CNN();
        int predict(const Mat& frame) override;
        int predict(const Mat& frame, float &prob, float prob_thresh);
        void train() override;

    private:
        void softmax(Mat& mat);
        // string model = "../src/utils/tools/armor.bak.onnx";
        string model = "../src/utils/tools/model_best.onnx";
        dnn::Net net;
    };
}


#endif //AUTOAIM_CLASSIFIER_H
