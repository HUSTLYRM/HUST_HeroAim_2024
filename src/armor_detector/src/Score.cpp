/**
 * @file
 * @author
 * @brief
 *
 * @note
 *
 */
#include "Score.h"

using namespace cv;

namespace ly
{

    void Score::trainSVM(Mat sampleMat, Mat labelMat) {
        // create SVM
        svm = ml::SVM::create();
        // set params
        svm->setType(ml::SVM::C_SVC);
        svm->setKernel(ml::SVM::LINEAR);
        svm->setC(1);

        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 5000, 1e-7));
        // train
        Ptr<ml::TrainData> trainData = ml::TrainData::create(sampleMat, ml::ROW_SAMPLE, labelMat);
        svm->train(trainData);
        svm->save(svmModel);
    }
}


