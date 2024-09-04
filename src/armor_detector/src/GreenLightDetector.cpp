//
// Created by MicDZ on 24-7-13.
//

//
// Created by MicDZ on 2024/4/7.
//
#include "GreenLightDetector.hpp"

GreenLightDetector::GreenLightDetector() {
    // Load the Inference Engine
//    ie = ov::Core();
//
//    // Load the network using OpenVINO
//    compiled_model = ie.compile_model(model_path, "CPU");
//    infer_request = compiled_model.create_infer_request();
//    GreenLightDetector(this->model_path);
    ie = ov::Core();
    compiled_model = ie.compile_model(this->model_path, "CPU",ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    infer_request = compiled_model.create_infer_request();
}
GreenLightDetector::GreenLightDetector(string model_path) {
    // Load the Inference Engine
    ie = ov::Core();
    compiled_model = ie.compile_model(model_path, "CPU",ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

    infer_request = compiled_model.create_infer_request();
}

GreenLightDetector::~GreenLightDetector() {
}

Rect2f GreenLightDetector::getROI(cv::Mat img, bbox result) {

    float x1 = result.x1;
    float y1 = result.y1;
    float x2 = result.x2;
    float y2 = result.y2;
    float width = x2 - x1;
    float height = y2 - y1;

    return Rect2f(x1, y1, width, height);
}

void GreenLightDetector::visualizeResult(const Mat &img, bbox result) {
    rectangle(img, Point(result.x1, result.y1), Point(result.x2, result.y2), Scalar(0, 255, 0), 2);
    // labelname
    std::string label = class_names[result.class_id] + ":" + std::to_string(result.score).substr(0, 4);
    Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
    // conf
    Rect2f textBox(result.x1, result.y1 - 15, textSize.width, textSize.height+5);
    cv::rectangle(img, textBox, Scalar(0, 255, 0), FILLED);
    putText(img, label, Point(result.x1, result.y1 - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
}

Mat GreenLightDetector::letterbox(const cv::Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect2f(0, 0, col, row)));
    return result;
}

void GreenLightDetector::detect(const Mat img, vector<Rect2f> &rois, Mat &debugImg, double score_threshold) {
    // Preprocess the image
    // 测预处理时间

    Mat letterbox_img = letterbox(img);
    float scale = letterbox_img.size[0] / IMG_SIZE;
    Mat blob = dnn::blobFromImage(letterbox_img, 1.0 / 255.0, Size(IMG_SIZE, IMG_SIZE), Scalar(), true);
    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input

    auto input_port = compiled_model.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);


    // -------- Step 6. Start inference --------
    infer_request.infer();
    // -------- Step 7. Get the inference result --------
    output_tensor = infer_request.get_output_tensor(0);
    output_shape = output_tensor.get_shape();
    auto rows = output_shape[2];        //8400
    auto dimensions = output_shape[1];  //84: box[cx, cy, w, h]+80 classes scores

    // -------- Step 8. Postprocess the result --------
    auto* data = output_tensor.data<float>();
    Mat output_buffer(dimensions, rows, CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,84]

    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<Rect2f> boxes2f;
    // Figure out the bbox, class_id and class_score
    int outputBufferRows = output_buffer.rows;
    for (int i = 0; i < outputBufferRows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, dimensions);
        Point class_id;
        double maxClassScore;
        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > score_threshold) {
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);
            float left = float((cx - 0.5 * w) * scale);
            float top = float((cy - 0.5 * h) * scale);
            float width = float(w * scale);
            float height = float(h * scale);
            boxes.push_back(Rect(left, top, width, height));
            boxes2f.push_back(Rect2f(left, top, width, height));
        }
    }
    //NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);

    Mat draw_img = img.clone();
    bbox result;
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        result.x1 = boxes2f[index].tl().x;
        result.y1 = boxes2f[index].tl().y; //top left
        result.x2 = boxes2f[index].br().x;
        result.y2 = boxes2f[index].br().y; // bottom right
        result.class_id = class_ids[index];
        result.score = class_scores[index];
//         visualizeResult(draw_img, result);
        Rect2f item;
        item = getROI(img, result);

//        cout<<"color id: "<<result.class_id<<endl;
        rois.emplace_back(item);
    }
    // 画出roi
    for (auto roi:rois) {
        rectangle(debugImg, roi, Scalar(255, 255, 0), 2);
    }
//    imshow("result", draw_img);
//    waitKey(0);
}