//
// Created by zhiyu on 2021/9/11.
//

#include "trainSVM.h"

namespace fs = boost::filesystem;
using namespace ly;

int main(int argc, char** argv){
    auto Score_ = new Score();

    vector<vector<uchar>> samples;
    vector<int> labels;
    int id = 0;

    regex dir_regex("[0-9]+");

    fs::path root(picRoot);
    if(!fs::exists(root)){
        cout << "directory not exits." << endl;
        return -1;
    }

    Mat kernel = getStructuringElement(0, Size(3, 1));
    fs::directory_iterator root_iter(root), end_iter;
    
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4.);
    clahe->setTilesGridSize(Size(8, 8));

    float gamma = 1/2.5;
    unordered_map<int, float> gamma_table;
    for (int i = 0; i < 256; ++i)
    {
        gamma_table[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
    }

                                    // block    // stride   // cell
    HOGDescriptor hog(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 10);
    int feature;

    for(;root_iter != end_iter; root_iter++){
        if(fs::is_directory(root_iter->status()) && regex_match(root_iter->path().filename().string(), dir_regex)){
            fs::path picPath(root_iter->path().string());
            fs::directory_iterator picIter(picPath), endIter;
            string label = root_iter->path().filename().string();
            cout << label << endl;
            for(; picIter != endIter; picIter++){

                cv::Mat pic = cv::imread(picIter->path().string(), 1);
                if(pic.empty()) continue;

                for(int ii=0;ii<32;ii++){
                    for(int jj=0;jj<32;jj++){
                        for(int c=0;c<3;c++){
                            pic.at<Vec3b>(ii, jj)[c] = (uchar)gamma_table[pic.at<Vec3b>(ii, jj)[c]];
                        }
                    }
                }
                cv::cvtColor(pic, pic, COLOR_BGR2GRAY);
                // cout << picIter->path() << endl;
                cv::medianBlur(pic, pic, 5);
                // cv::GaussianBlur(pic, pic, Size(5, 5), 1, 1);
                // clahe->apply(pic, pic);
                cv::equalizeHist(pic, pic);

                // cv::threshold(pic, pic, 0, 255, THRESH_OTSU);
                
                vector<float> desc;
                hog.compute(pic, desc, Size(8, 8), Size(0, 0));
                feature = desc.size();
                Mat x(1, feature, CV_32FC1);
                for(int i=0;i<desc.size();i++){
                    x.at<float>(0, i) = desc[i] * 100;
                }
                
                // cv::MatIterator_<uchar> matIter = pic.begin<uchar>(), end = pic.end<uchar>();
                // vector<uchar> picVec;
                // for(; matIter != end; matIter++){
                //     picVec.emplace_back((uchar)*matIter);
                // }

                cv::MatIterator_<float> matIter = x.begin<float>(), end = x.end<float>();
                vector<uchar> picVec;
                for(; matIter != end; matIter++){
                    picVec.emplace_back((uchar)*matIter);
                    // cout << *matIter << endl;
                }
                samples.emplace_back(picVec);
                labels.emplace_back(stoi(label));
//                cout << stoi(label) << " ";
                id++;
            }
//            cout << endl;
        }
    }
    int sampleNum = samples.size();
    int cols = feature;
    cv::Mat sampleMat(sampleNum, cols, CV_32FC1);
    cv::Mat labelMat(sampleNum, 1, CV_32SC1);

    for(int i=0;i<sampleNum;i++){
        for(int j=0;j<cols;j++){
            sampleMat.at<float>(i, j) = (float)samples[i][j];
        }
        labelMat.at<int>(i, 0) = (int)labels[i];
//        cout << labels[i] << " ";
//        cout << labelMat.at<int>(i, 0) << " ";
    }

//    PCA pca = PCA(sampleMat, Mat(), PCA::DATA_AS_ROW, 20);
//    FileStorage storage("../src/utils/tools/pca.xml", FileStorage::WRITE);
//    pca.write(storage);
//    storage.release();
//    sampleMat = pca.project(sampleMat);
    sampleMat.convertTo(sampleMat, CV_32FC1);
    labelMat.convertTo(labelMat, CV_32SC1);
//    cout << sampleMat;
//    cout << labelMat;

    Score_->trainSVM(sampleMat, labelMat);
}