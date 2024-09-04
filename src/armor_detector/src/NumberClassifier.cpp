#include "NumberClassifier.h"
#define svm 0
#define mlp 1
#define resnet 2
namespace ly
{
    NumberClassifier::NumberClassifier(int mode)
    {

        if (mode == svm)
        {
            mode_ = mode;
            loadModel("./model/svm_numbers_rbf.xml");
            initHog();
            calcGammaTable(CameraParam::gamma);
        }
    }

    NumberClassifier::~NumberClassifier()
    {
        delete hog_;
    }

    void NumberClassifier::initHog()
    {
        // 窗口大小,块大小，块步长，cell，180度分为几个区间
        hog_ = new HOGDescriptor(cv::Size(32, 32), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 16);
    }

    void NumberClassifier::calcGammaTable(float gamma)
    {
        for (int i = 0; i < 256; ++i)
        {
            gamma_table[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
        }
        lut_table = Mat(1, 256, CV_8UC1, gamma_table);
    }

    // 加载模型
    bool NumberClassifier::loadModel(const std::string &model_path)
    {
        number_svm_model = cv::ml::SVM::load(model_path);
        if (number_svm_model.empty())
        {
            std::cerr << "Open NUMBER svm model ERROR" << std::endl;
            is_model_set = false; // 模型不可用
        }
        else
        {
            is_model_set = true;
        }
        return is_model_set;
    }

    // 使用形态学找灯条时warp会不准，进行后处理的数据增强(不支持mlp模式)
    std::pair<bool, vector<cv::Mat>> NumberClassifier::affineMultiNumber(Mat &frame, const std::vector<cv::Point2f> &corners)
    {
        if (mode_ == svm || mode_ == resnet)
        {
            is_get_roi = false;

            static float classify_width_ratio = 0.2f;
            static float classify_height_ratio[5] = {0.5f, 0.4f, 0.6f};

            static float height_zoom_ratio[5] = {0.8, 1.0, 1.2};

            vector<cv::Mat> number_roi_multi;

            // 竖直缩放
            for (int k = 0; k < 3; k++)
            {
                for (int i_ = 0; i_ < 3; i_++)
                {
                    // 求解完全包围这个框的最大矩形
                    cv::Point2f correct_points[4];
                    cv::Point2f width_vec = (corners[1] - corners[0] + corners[2] - corners[3]) / 2;
                    cv::Point2f height_vec = (corners[3] - corners[0] + corners[2] - corners[1]) / 2;

                    correct_points[0] = (corners[0] + corners[3]) / 2 - 0.5 * (height_vec)*height_zoom_ratio[k];
                    correct_points[3] = (corners[0] + corners[3]) / 2 + 0.5 * (height_vec)*height_zoom_ratio[k];
                    correct_points[1] = (corners[1] + corners[2]) / 2 - 0.5 * (height_vec)*height_zoom_ratio[k];
                    correct_points[2] = (corners[1] + corners[2]) / 2 + 0.5 * (height_vec)*height_zoom_ratio[k];

                    correct_points[0] = corners[0] + classify_width_ratio * width_vec - classify_height_ratio[i_] * height_vec;
                    correct_points[1] = corners[1] - classify_width_ratio * width_vec - classify_height_ratio[i_] * height_vec;
                    correct_points[2] = corners[2] - classify_width_ratio * width_vec + classify_height_ratio[i_] * height_vec;
                    correct_points[3] = corners[3] + classify_width_ratio * width_vec + classify_height_ratio[i_] * height_vec;

                    int width = getDistance(correct_points[0], correct_points[1]);
                    int height = getDistance(correct_points[1], correct_points[2]);
                    cv::Point2f min_point = cv::Point2f(9999.0f, 9999.0f);
                    cv::Point2f max_point = cv::Point2f(0.0f, 0.0f);
                    for (int i = 0; i < 4; i++)
                    {
                        min_point.x = min_point.x < correct_points[i].x ? min_point.x : correct_points[i].x;
                        min_point.y = min_point.y < correct_points[i].y ? min_point.y : correct_points[i].y;
                        max_point.x = max_point.x > correct_points[i].x ? max_point.x : correct_points[i].x;
                        max_point.y = max_point.y > correct_points[i].y ? max_point.y : correct_points[i].y;
                    }
                    min_point.x = MAX(min_point.x, 0);
                    min_point.y = MAX(min_point.y, 0);
                    max_point.x = MIN(max_point.x, frame.cols);
                    max_point.y = MIN(max_point.y, frame.rows);

                    // 截取
                    cv::Mat m_number_roi = frame(cv::Rect(min_point, max_point));

                    for (int i = 0; i < 4; i++)
                    {
                        correct_points[i] -= min_point;
                    }

                    // 制作重映射对应点
                    cv::Point2f remap_points[4];
                    remap_points[0] = cv::Point2f(0, 0);
                    remap_points[1] = cv::Point2f((int)width, 0);
                    remap_points[2] = cv::Point2f((int)width, (int)height);
                    remap_points[3] = cv::Point2f(0, (int)height);

                    // 进行重映射
                    cv::Mat trans_matrix = cv::getPerspectiveTransform(correct_points, remap_points);
                    cv::Mat output_roi;
                    output_roi.create(cv::Size((int)width, (int)height), CV_8UC3);

                    if (m_number_roi.empty() || output_roi.empty())
                    {
                        continue;
                    }

                    cv::warpPerspective(m_number_roi, output_roi, trans_matrix, output_roi.size());

                    cv::Mat number_roi_temp;
                    // //从重映射中取得目标图像
                    cv::resize(output_roi, number_roi_temp, cv::Size(32, 32)); // 根据训练的数据大小来判断大小

                    number_roi_multi.push_back(number_roi_temp);
                }
            }
            vector<cv::Mat> number_roi_multi_temp = number_roi_multi;

            // 旋转变换
            cv::Mat R_0 = getRotationMatrix2D(Point2f(16, 16), 0, 1.0);
            cv::Mat R_1 = getRotationMatrix2D(Point2f(16, 16), -10, 1.0);
            cv::Mat R_2 = getRotationMatrix2D(Point2f(16, 16), 10, 1.0);
            cv::Mat R_3 = getRotationMatrix2D(Point2f(16, 16), -15, 1.0);
            cv::Mat R_4 = getRotationMatrix2D(Point2f(16, 16), 15, 1.0);
            for (int j = 0; j < number_roi_multi_temp.size(); j++)
            {
                cv::Mat number_roi_temp;
                warpAffine(number_roi_multi_temp.at(j), number_roi_temp, R_0, Size(32, 32), INTER_LINEAR, 0);
                number_roi_multi.push_back(number_roi_temp);
                warpAffine(number_roi_multi_temp.at(j), number_roi_temp, R_1, Size(32, 32), INTER_LINEAR, 0);
                number_roi_multi.push_back(number_roi_temp);
                warpAffine(number_roi_multi_temp.at(j), number_roi_temp, R_2, Size(32, 32), INTER_LINEAR, 0);
                number_roi_multi.push_back(number_roi_temp);
                warpAffine(number_roi_multi_temp.at(j), number_roi_temp, R_3, Size(32, 32), INTER_LINEAR, 0);
                number_roi_multi.push_back(number_roi_temp);
                warpAffine(number_roi_multi_temp.at(j), number_roi_temp, R_4, Size(32, 32), INTER_LINEAR, 0);
                number_roi_multi.push_back(number_roi_temp);
            }
            is_get_roi = true;

            // imshow("roi_number" + to_string(numbers_), number_roi);
            return std::pair<bool, vector<cv::Mat>>(is_get_roi, number_roi_multi);
        }

        else // mlp模式，暂不进行适配
        {
            return std::pair<bool, vector<cv::Mat>>(false, vector<cv::Mat>());
        }
    }

    bool NumberClassifier::affineNumber(Mat &frame, const std::vector<cv::Point2f> &corners)
    {
        if (mode_ == svm || mode_ == resnet)
        {
            is_get_roi = false;
            static float classify_width_ratio = 0.2f;
            static float classify_height_ratio = 0.5f;

            // 求解完全包围这个框的最大矩形
            cv::Point2f correct_points[4];
            cv::Point2f width_vec = (corners[1] - corners[0] + corners[2] - corners[3]) / 2;
            cv::Point2f height_vec = (corners[3] - corners[0] + corners[2] - corners[1]) / 2;
            correct_points[0] = corners[0] + classify_width_ratio * width_vec - classify_height_ratio * height_vec;
            correct_points[1] = corners[1] - classify_width_ratio * width_vec - classify_height_ratio * height_vec;
            correct_points[2] = corners[2] - classify_width_ratio * width_vec + classify_height_ratio * height_vec;
            correct_points[3] = corners[3] + classify_width_ratio * width_vec + classify_height_ratio * height_vec;

            int width = getDistance(correct_points[0], correct_points[1]);
            int height = getDistance(correct_points[1], correct_points[2]);
            cv::Point2f min_point = cv::Point2f(9999.0f, 9999.0f);
            cv::Point2f max_point = cv::Point2f(0.0f, 0.0f);
            for (int i = 0; i < 4; i++)
            {
                min_point.x = min_point.x < correct_points[i].x ? min_point.x : correct_points[i].x;
                min_point.y = min_point.y < correct_points[i].y ? min_point.y : correct_points[i].y;
                max_point.x = max_point.x > correct_points[i].x ? max_point.x : correct_points[i].x;
                max_point.y = max_point.y > correct_points[i].y ? max_point.y : correct_points[i].y;
            }
            min_point.x = MAX(min_point.x, 0);
            min_point.y = MAX(min_point.y, 0);
            max_point.x = MIN(max_point.x, frame.cols);
            max_point.y = MIN(max_point.y, frame.rows);

            // 截取
            cv::Mat m_number_roi = frame(cv::Rect(min_point, max_point));

            for (int i = 0; i < 4; i++)
            {
                correct_points[i] -= min_point;
            }

            // 制作重映射对应点
            cv::Point2f remap_points[4];
            remap_points[0] = cv::Point2f(0, 0);
            remap_points[1] = cv::Point2f((int)width, 0);
            remap_points[2] = cv::Point2f((int)width, (int)height);
            remap_points[3] = cv::Point2f(0, (int)height);

            // 进行重映射
            cv::Mat trans_matrix = cv::getPerspectiveTransform(correct_points, remap_points);
            cv::Mat output_roi;
            output_roi.create(cv::Size((int)width, (int)height), CV_8UC3);

            if (m_number_roi.empty() || output_roi.empty())
            {
                return false;
            }
            cv::warpPerspective(m_number_roi, output_roi, trans_matrix, output_roi.size());

            // //从重映射中取得目标图像
            cv::resize(output_roi, number_roi, cv::Size(32, 32)); // 根据训练的数据大小来判断大小
            is_get_roi = true;

            // imshow("roi_number" + to_string(numbers_), number_roi);
            return is_get_roi;
        }
        else if (mode_ == mlp)
        {
            // Light length in image
            const int light_length = 12;
            // Image size after warp
            const int warp_height = 28;
            const int warp_width = 32;
            // Number ROI size
            const cv::Size roi_size(20, 28);

            cv::Point2f lights_vertices[4] = {
                corners[3], corners[0], corners[1], corners[2]};

            const int top_light_y = (warp_height - light_length) / 2 - 1;
            const int bottom_light_y = top_light_y + light_length;
            cv::Point2f target_vertices[4] = {
                cv::Point(0, bottom_light_y),
                cv::Point(0, top_light_y),
                cv::Point(warp_width - 1, top_light_y),
                cv::Point(warp_width - 1, bottom_light_y),
            };
            cv::Mat number_image;
            auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
            cv::warpPerspective(frame, number_image, rotation_matrix, cv::Size(warp_width, warp_height));
            number_roi = number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));
            if (number_roi.empty())
            {
                return false;
            }
            // Binarize
            cv::cvtColor(number_roi, number_roi, cv::COLOR_RGB2GRAY);
            cv::threshold(number_roi, number_roi, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

            return true;
        }
    }

    float NumberClassifier::getDistance(const cv::Point2f &point_1, const cv::Point2f &point_2)
    {
        float x = (point_1 - point_2).x;
        float y = (point_1 - point_2).y;
        return sqrt(x * x + y * y);
    }

    void NumberClassifier::AutoGamma(const cv::Mat &img, cv::Mat &out)
    {
        const int channels = img.channels();
        const int type = img.type();
        /*debug*/
        assert(type == CV_8UC1 || type == CV_8UC3);

        auto mean = cv::mean(img); // 均值
        mean[0] = std::log10(0.4) / std::log10(mean[0] / 255);

        if (channels == 3) // 3channels
        {
            mean[1] = std::log10(0.4) / std::log10(mean[1] / 255);
            mean[2] = std::log10(0.4) / std::log10(mean[2] / 255);

            float mean_end = 0;
            for (int i = 0; i < 3; ++i)
                mean_end += mean[i];

            mean_end /= 3.0;
            for (int i = 0; i < 3; ++i)
                mean[i] = mean_end;
        }
        /*gamma_table*/
        cv::Mat lut(1, 256, img.type());

        if (channels == 1)
        {
            for (int i = 0; i < 256; i++)
            { /*[0,1]*/
                float Y = i * 1.0f / 255.0;
                Y = std::pow(Y, mean[0]);
                lut.at<unsigned char>(0, i) = cv::saturate_cast<unsigned char>(Y * 255);
            }
        }
        else
        {
            for (int i = 0; i < 256; ++i)
            {
                float Y = i * 1.0f / 255.0;
                auto B = cv::saturate_cast<unsigned char>(std::pow(Y, mean[0]) * 255);
                auto G = cv::saturate_cast<unsigned char>(std::pow(Y, mean[1]) * 255);
                auto R = cv::saturate_cast<unsigned char>(std::pow(Y, mean[2]) * 255);

                lut.at<cv::Vec3b>(0, i) = cv::Vec3b(B, G, R);
            }
        }
        cv::LUT(img, lut, out);
    }

    cv::Mat NumberClassifier::AutoGammaCorrect(const cv::Mat image)
    {
        cv::Mat img;
        cv::Mat dst;
        image.copyTo(dst);
        image.copyTo(img);

        cv::cvtColor(img, img, COLOR_BGR2HSV);
        cv::cvtColor(dst, dst, COLOR_BGR2GRAY);
        const int type = img.type();
        // assert(type == CV_8UC1 || type == CV_8UC3);

        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);
        cv::Mat V = channels[2];
        V.convertTo(V, CV_32F);
        double max_v;
        cv::minMaxIdx(V, 0, &max_v);
        V /= 255.0;

        cv::Mat Mean, Sigma;
        cv::meanStdDev(V, Mean, Sigma);

        double mu, sigma; /*均值and方差*/
        bool High_contrast = false;
        bool High_bright = false;
        mu = Mean.at<double>(0);
        sigma = Sigma.at<double>(0);
        // std::cout<<"mu = "<<mu<<"sigma = "<<sigma<<std::endl;

        if (4 * sigma > 0.3333)
        {
            High_contrast = true;
            //    std::cout<<"High_con"<<std::endl;
        }
        if (mu > 0.65)
        {
            High_bright = true;
            //    std::cout<<"High_bri"<<std::endl;
        }
        double gamma, c, Heaviside;

        if (High_contrast)
        {
            gamma = std::exp((1 - (mu + sigma)) / 2.0);
        }
        else
        {
            gamma = -std::log(sigma) / std::log(2);
        }
        // std::cout << gamma << std::endl;

        return pixel(dst, gamma, mu, sigma, High_bright);
    }

    cv::Mat NumberClassifier::pixel(const cv::Mat image, double gamma, double mu, double sigma, int flag)
    {
        double K;
        cv::Mat img;
        image.copyTo(img);
        int rows = image.rows;
        int cols = image.cols;
        int channels = image.channels();
        if (flag)
        {
            for (int i = 0; i < cols; ++i)
            {
                for (int j = 0; j < rows; ++j)
                {
                    if (channels == 3)
                        for (int k = 0; k < channels; ++k)
                        {
                            float pix = float(img.at<Vec3b>(i, j)[k]) / 255.0;
                            img.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(std::pow(pix, gamma) * 255.0f);
                        }
                    else
                    {
                        float pix = float(img.at<uchar>(i, j)) / 255.0;
                        img.at<uchar>(i, j) = saturate_cast<uchar>(std::pow(pix, gamma) * 255.0f);
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < cols; ++i)
            {
                for (int j = 0; j < rows; ++j)
                {
                    if (channels == 3)
                        for (int k = 0; k < channels; ++k)
                        {
                            float pix = float(img.at<Vec3b>(i, j)[k]) / 255.0;
                            double t = std::pow(mu, gamma);
                            K = std::pow(pix, gamma) + (1 - std::pow(pix, gamma)) * t;
                            img.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(std::pow(pix, gamma) / K * 255.0f);
                        }
                    else
                    {
                        float pix = float(img.at<uchar>(i, j)) / 255.0;
                        double t = std::pow(mu, gamma);
                        K = std::pow(pix, gamma) + (1 - std::pow(pix, gamma)) * t;
                        img.at<uchar>(i, j) = saturate_cast<uchar>(std::pow(pix, gamma) / K * 255.0f);
                    }
                }
            }
        }
        return img;
    }

    std::pair<int, double> NumberClassifier::predict(Mat &frame, const std::vector<cv::Point2f> &corners, int find_mode)
    {

        if (find_mode == 1) // find mode = thresh
        {
            if (!affineNumber(frame, corners) || number_roi.empty())
            {
                return pair<int, double>(-1, 0);
            }

            if (mode_ == svm || mode_ == resnet)
            {
                //  不保存图片记得注释掉

                // cvtColor(number_roi, number_roi, COLOR_BGR2GRAY);
                //   medianBlur(number_roi, number_roi, 3);

                // equalizeHist(number_roi, number_roi);

                // 自适应gamma
                // AutoGamma(number_roi, number_roi);

                // 自适应gamma PLUS
                number_roi = AutoGammaCorrect(number_roi);

                // 普通gamma
                // cv::LUT(number_roi, lut_table, number_roi);

                // imshow("roi_number", number_roi);
                if (mode_ == svm)
                {

                    std::vector<float> hog_descriptors;
                    // 对图片提取hog描述子存在hog_descriptors中，hog描述子是向量，不是矩阵
                    hog_->compute(number_roi, hog_descriptors, Size(8, 8), Size(0, 0));
                    size_t size = hog_descriptors.size();
                    cv::Mat descriptors_mat(1, size, CV_32FC1); // 行向量
                    for (size_t i = 0; i < hog_descriptors.size(); ++i)
                    {
                        descriptors_mat.at<float>(0, i) = hog_descriptors[i] * 100;
                    }



                    // 把提取的本张图片的hog描述子放进svm预测器中进行预测
                    number_svm_model->predict(descriptors_mat, class_);
                    saveImage();
                    // if ((int)class_.at<float>(0) > 0)
                    // {
                    //     showNumber();
                    // }

                    return pair<int, double>((int)class_.at<float>(0), 1);
                }
                else if (mode_ == resnet)
                {
                    Mat blob;
                    dnn::blobFromImage(number_roi, blob, 1, Size(32, 32), Scalar(0, 0, 0), false, false, CV_32FC1);
                    net_.setInput(blob);

                    Mat predict = net_.forward();

                    softmax(predict);

                    int class_ = 0;
                    float prob = 0;
                    for (int i = 0; i < 9; i++)
                    {
                        if (predict.at<float>(i) > prob)
                        {
                            prob = predict.at<float>(i);
                            class_ = i;
                        }
                    }
                    // DLOG(WARNING)<<"DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"<<endl;
                    saveImage();
                    return pair<int, double>(class_, prob);
                }
            }

            else if (mode_ == mlp)
            {
                number_roi = number_roi / 255.0;
                cv::Mat blob;
                cv::dnn::blobFromImage(number_roi, blob, 1., cv::Size(28, 20));

                // Set the input blob for the neural network
                net_.setInput(blob);

                // Forward pass the image blob through the model
                cv::Mat outputs = net_.forward();

                // Do softmax
                float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
                cv::Mat softmax_prob;
                cv::exp(outputs - max_prob, softmax_prob);
                float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
                softmax_prob /= sum;

                double confidence;
                cv::Point class_id_point;
                minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
                int label_id = class_id_point.x;

                return pair<int, double>(label_id, confidence);
            }
        }
        else // find mode = grad
        {
            std::pair<bool, std::vector<cv::Mat>> affine_result = affineMultiNumber(frame, corners);
            bool affine_success = affine_result.first;
            std::vector<cv::Mat> rois = affine_result.second;

            if (!affine_success)
            {
                return pair<int, double>(-1, 0);
            }

            std::vector<int> class_vec;
            std::vector<float> prob_vec;

            for (int i = 0; i < rois.size(); i++)
            {
                if (mode_ == svm || mode_ == resnet)
                {
                    // cvtColor(rois[i], rois[i], COLOR_BGR2GRAY);
                    //  medianBlur(number_roi, number_roi, 3);

                    // equalizeHist(rois[i], rois[i]);

                    // 自适应gamma
                    // AutoGamma(rois[i], rois[i]);

                    // 自适应gamma PLUS
                    rois[i] = AutoGammaCorrect(rois[i]);

                    // 普通gamma
                    // cv::LUT(rois[i], lut_table, rois[i]);

                    // imshow("roi_number" + to_string(numbers_),rois[i]);

                    if (mode_ == svm)
                    {
                        std::vector<float> hog_descriptors;
                        // 对图片提取hog描述子存在hog_descriptors中，hog描述子是向量，不是矩阵
                        hog_->compute(rois[i], hog_descriptors);
                        size_t size = hog_descriptors.size();
                        cv::Mat descriptors_mat(1, size, CV_32FC1); // 行向量
                        for (size_t i = 0; i < hog_descriptors.size(); ++i)
                        {
                            descriptors_mat.at<float>(0, i) = hog_descriptors[i] * 100;
                        }

                        // 把提取的本张图片的hog描述子放进svm预测器中进行预测
                        number_svm_model->predict(descriptors_mat, class_);

                        class_vec.push_back((int)class_.at<float>(0));
                        prob_vec.push_back(1);
                    }
                    else if (mode_ == resnet)
                    {
                        // imshow("roi_number", number_roi);
                        // DLOG(WARNING)<<"IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"<<endl;
                        Mat blob;
                        dnn::blobFromImage(rois[i], blob, 1, Size(32, 32), Scalar(0, 0, 0), false, false, CV_32FC1);
                        net_.setInput(blob);

                        Mat predict = net_.forward();

                        softmax(predict);

                        int class_ = 0;
                        float prob = 0;
                        for (int i = 0; i < 9; i++)
                        {
                            if (predict.at<float>(i) > prob)
                            {
                                prob = predict.at<float>(i);
                                class_ = i;
                            }
                        }
                        class_vec.push_back(class_);
                        prob_vec.push_back(prob);
                    }
                }
                else if (mode_ == mlp) // 不使用mlp
                {
                    return pair<int, double>(-1, 0);
                }
            }

            for (int k = 0; k < class_vec.size(); k++)
            {
                if (class_vec.at(k) > 0 && prob_vec.at(k) > 0.5)
                {
                    return pair<int, double>(class_vec.at(k), prob_vec.at(k));
                }
            }

            return pair<int, double>(-1, 0);
        }
    }
    void NumberClassifier::showNumber()
    {

    }
    void NumberClassifier::showNumberStr(cv::Mat &drawing, int id, double conf, cv::Rect rect)
    {

    }
    void NumberClassifier::saveImage()
    {

    }
    void NumberClassifier::softmax(Mat &mat)
    {
        float denominator = 0;
        for (int i = 0; i < 9; i++)
        {
            denominator += exp(mat.at<float>(i));
        }
        for (int i = 0; i < 9; i++)
        {
            mat.at<float>(i) = exp(mat.at<float>(i)) / denominator;
        }
    }
}