//
// Created by zhiyu on 2021/8/30.
//

#include "Tracker.h"

using namespace ly;

ly::Tracker::Tracker() {
    this->_init = false;
    this->_state = FOUND;
    this->_learning_rate = 0.125;

    this->_image_size.width = 0;
    this->_image_size.height = 0;

    this->_PSR_mask = 11;
    this->_PSR_ratio[0] = 5;
    this->_PSR_ratio[1] = 9;

    this->_eps = true;
}

void ly::Tracker::initTracker(const Mat &input_image, Point2i top_left, Point2i bottom_right) {
    initTracker(input_image,
                  Rect(top_left.x, top_left.y,
                        bottom_right.x - top_left.x,
                        bottom_right.y - top_left.y));
}

void ly::Tracker::initTracker(const Mat &input_image, Rect input_rect) {
    if(this->_init) return;

    if(input_image.empty()){
        DLOG(ERROR) << "input image is empty! ";
        return;
    }

    this->_image_size.width = input_image.cols;
    this->_image_size.height = input_image.rows;

    input_image(input_rect).copyTo(prev_image.frame);

    this->prev_image.cols = this->prev_image.frame.cols;
    this->prev_image.rows = this->prev_image.frame.rows;

    calcDFT(prev_image, true);
    setRoi(input_rect);

    initFilter();
    this->_init = true;
}

void ly::Tracker::calcDFT(Image &input_image, bool preprocess) {
    Mat res = this->calcDFT(input_image.frame, preprocess);

    input_image.image_sepctrum = res;
    input_image.optimal_dft_cols = res.cols;
    input_image.optimal_dft_rows = res.rows;
}

/**
 * padding + preprocess + dft + crop
 * @param input_image
 * @param preprocess
 * @return
 */
Mat ly::Tracker::calcDFT(const Mat &input_image, bool preprocess) {

    Mat gray_padded, complex;

    int x = input_image.rows;
    int y = input_image.cols;

    // 获取最优DFT尺寸
    int i = getOptimalDFTSize(x);
    int j = getOptimalDFTSize(y);

    // zero padding
    copyMakeBorder(input_image, gray_padded,
                   0, i-x, 0, j-y,
                   BORDER_CONSTANT,
                   Scalar::all(0));

    input_image.copyTo(gray_padded);

    if(gray_padded.channels() > 1) cvtColor(gray_padded, gray_padded, COLOR_BGR2GRAY);
    gray_padded.convertTo(gray_padded, CV_32F);

    if(preprocess){
        normalize(gray_padded, gray_padded, 0, 1, NORM_MINMAX);

        gray_padded += Scalar::all(0);
        log(gray_padded, gray_padded);

        Scalar mean, stddev;
        meanStdDev(gray_padded, mean, stddev);
        gray_padded -= mean.val[0];

        Mat tmp;
        multiply(gray_padded, gray_padded, tmp);
        Scalar  sum_ = sum(tmp);
        gray_padded /= sum_.val[0];

        if(this->_hanning_window.empty() || gray_padded.size() != this->_hanning_window.size()){
            Mat hanning_window;
            createHanningWindow(hanning_window, gray_padded.size(), CV_32F);
            hanning_window.copyTo(this->_hanning_window);
        }

        multiply(gray_padded, this->_hanning_window, gray_padded);
    }

    dft(gray_padded, complex, DFT_COMPLEX_OUTPUT);

    // 裁剪奇数行
    complex = complex(Rect(0, 0, complex.cols & -2, complex.rows & -2));

    return complex;
}

void ly::Tracker::setRoi(const Rect& input_roi) {

    this->curr_roi.roi = input_roi;
    this->curr_roi.roi_center.x = round(input_roi.width/2);
    this->curr_roi.roi_center.y = round(input_roi.height/2);

}

void ly::Tracker::initFilter() {

    Mat mask_gauss = Mat::zeros(this->prev_image.frame.size(), CV_32F);
    maskDesiredG(mask_gauss, round(this->curr_roi.roi.width/2),
                 round(this->curr_roi.roi.height/2));

    Mat tmp_desired_g = calcDFT(mask_gauss, false);
    tmp_desired_g.copyTo(this->prev_image.filter_output);

    Mat tmp_FG = Mat::zeros(this->prev_image.optimal_dft_rows,
                            this->prev_image.optimal_dft_cols,
                            this->prev_image.image_sepctrum.type());
    Mat tmp_FF = Mat::zeros(this->prev_image.optimal_dft_rows,
                            this->prev_image.optimal_dft_cols,
                            this->prev_image.image_sepctrum.type());
    Mat tmp_image_dft, num, dem;
    tmp_image_dft = this->prev_image.image_sepctrum;
    mulSpectrums(tmp_desired_g, tmp_image_dft, num, 0, true);
    tmp_FG += num;
    mulSpectrums(tmp_image_dft, tmp_image_dft, dem, 0, true);
    tmp_FF += dem;

    Mat eps;
    if(this->_eps){
        eps = createEps(dem);
        dem += eps;
        tmp_FF += dem;
    }

    srand(time(nullptr));

    constexpr int N = 8;
    Mat affine_G, affine_image;
    for(int i=0;i<N-1;i++){
        AffineTransform(mask_gauss, this->prev_image.frame, affine_G, affine_image);

        tmp_image_dft = calcDFT(affine_image, true);
        tmp_desired_g = calcDFT(affine_G, false);

        mulSpectrums(tmp_desired_g, tmp_image_dft, num, 0, true);
        tmp_FG += num;
        mulSpectrums(tmp_image_dft, tmp_image_dft, dem, 0, true);
        if(this->_eps){
            eps = createEps(dem);
            dem += eps;
        }
        tmp_FF += dem;
    }
    Mat filter;
    dftDiv(tmp_FG, tmp_FF, filter);
    filter.copyTo(this->_filter);
}

void ly::Tracker::maskDesiredG(Mat &output, int u_x, int u_y, double sigma, bool norm_energy) {

    sigma *= sigma; // sigma = 2
    // 二维高斯曲面
    for(int i=0;i<output.rows;i++){
        for(int j=0;j<output.cols;j++){
            output.at<float>(i, j) = 255 * exp(-(i-u_y)*(i-u_y)/(2*sigma)
                    -(j-u_x)*(j-u_x)/(2*sigma));
        }
    }

    if(norm_energy){
        Scalar sum_ = sum(output);
        output /= sum_.val[0];
    }
}

Mat ly::Tracker::createEps(const Mat &input_, double std) {

    Scalar mean, stddev;
    meanStdDev(input_, mean, stddev);

    Mat eps = Mat::zeros(input_.size(), input_.type());

    randn(eps, 0, std*(mean.val[0]));

    for(int x=0;x<eps.rows;x++){
        for(int y=0;y<eps.cols;y++){
            eps.at<Vec2f>(x, y)[1] = 0;
        }
    }

    eps.at<Vec2f>(0, 0)[0] = 0;
    eps.at<cv::Vec2f>(input_.rows-1,0)[0] = 0;
    eps.at<cv::Vec2f>(0,input_.cols-1)[0] = 0;
    eps.at<cv::Vec2f>(input_.rows-1,input_.cols-1)[0] = 0;

    return eps;
}

void ly::Tracker::AffineTransform(const Mat &input_image, const Mat &input_image2, Mat &aff_img, Mat &aff_img2)
{//Apply same randomly defined affine transform to both input matrice

    if (input_image.size() != input_image2.size())
    {
        DLOG(ERROR)<<"Error while computing affine transform !";
        return;
    }

    //output images
    aff_img = Mat::zeros(input_image.rows,input_image.cols,input_image.type());
    aff_img2 = Mat::zeros(input_image2.rows,input_image2.cols,input_image2.type());

    int cols = input_image.cols;
    int rows = input_image.rows;

    Point2f input_pts[3];
    Point2f output_pts[3];

    float pts0_r, pts0_c, pts1_r, pts1_c, pts2_r, pts2_c;

    Mat affine_tr(2,3,CV_32FC1);

    input_pts[0] = Point2f(0,0);
    input_pts[1] = Point2f(cols - 1, 0);
    input_pts[2] = Point2f(0, rows - 1);

    //Define affine transform 'intensity'
    pts0_r = rand() % 5; pts0_r /= 100;
    pts0_c = rand() % 5; pts0_c /= 100;

    pts1_r = rand() % 5; pts1_r /= 100;
    pts1_c = rand() % 5 + 95; pts1_c /= 100;

    pts2_r = rand() % 5 + 95; pts2_r /= 100;
    pts2_c = rand() % 5; pts2_c /= 100;

    output_pts[0] = Point2f(cols*pts0_c,rows*pts0_r);
    output_pts[1] = Point2f(cols*pts1_c,rows*pts1_r);
    output_pts[2] = Point2f(cols*pts2_c,rows*pts2_r);

    affine_tr = getAffineTransform( input_pts, output_pts );        //Get transformation matrix

    warpAffine( input_image, aff_img, affine_tr, aff_img.size() );  //Apply transformation matrix
    warpAffine( input_image2, aff_img2, affine_tr, aff_img2.size() );
}

void ly::Tracker::dftDiv(const Mat &dft_a, const Mat &dft_b, Mat &output_dft)
{//Compute complex divison

    assert (dft_a.size() == dft_b.size() && dft_a.type() == dft_b.type() &&
            dft_a.channels() == dft_b.channels() && dft_a.channels() == 2);

    Mat out_temp = Mat::zeros(dft_a.rows,dft_a.cols,dft_a.type());

    for (int x=0;x<dft_a.rows;x++)
    {
        for (int y=0;y<dft_a.cols;y++)
        {
            out_temp.at<cv::Vec2f>(x,y)[0] = ( (dft_a.at<cv::Vec2f>(x,y)[0] * dft_b.at<cv::Vec2f>(x,y)[0]) +
                                               (dft_a.at<cv::Vec2f>(x,y)[1] * dft_b.at<cv::Vec2f>(x,y)[1]) ) /
                                             ( (dft_b.at<cv::Vec2f>(x,y)[0] * dft_b.at<cv::Vec2f>(x,y)[0]) +
                                               (dft_b.at<cv::Vec2f>(x,y)[1] * dft_b.at<cv::Vec2f>(x,y)[1]) );

            out_temp.at<cv::Vec2f>(x,y)[1] = ( (dft_a.at<cv::Vec2f>(x,y)[1] * dft_b.at<cv::Vec2f>(x,y)[0]) -
                                               (dft_a.at<cv::Vec2f>(x,y)[0] * dft_b.at<cv::Vec2f>(x,y)[1]) ) /
                                             ( (dft_b.at<cv::Vec2f>(x,y)[0] * dft_b.at<cv::Vec2f>(x,y)[0]) +
                                               (dft_b.at<cv::Vec2f>(x,y)[1] * dft_b.at<cv::Vec2f>(x,y)[1]) );
        }
    }

    out_temp.copyTo(output_dft);
}

void ly::Tracker::track(const Mat &frame) {

    if(!this->_init) return;
    if(this->_filter.empty()){
        DLOG(ERROR) << "Error, must initialize tracker first! ";
        return;
    }

    Point2i new_location;
    frame(this->curr_roi.roi).copyTo(this->curr_image.frame);
    calcDFT(this->curr_image, true);
    new_location = performTrack();

    if(new_location.x >= 0 && new_location.y >= 0){
        this->_state = FOUND;
        update(new_location);
    } else{
        this->_state = OCCLUDE;
    }
}

Point2i ly::Tracker::performTrack() {
    Mat mat_correlation, idft_correlation;

    mulSpectrums(this->curr_image.image_sepctrum, this->_filter, mat_correlation, 0, false);
    dft(mat_correlation, idft_correlation,
        DFT_INVERSE | DFT_REAL_OUTPUT);

#ifdef DEBUG
    /****** begin show images *******/
    Mat coor_image;
    normalize(idft_correlation ,idft_correlation, 0, 255, NORM_MINMAX);
    idft_correlation.copyTo(coor_image);    // could resize here
    imshow("correlation image", coor_image);

    Mat filter_image;
    dft(this->_filter, filter_image,
        DFT_INVERSE | DFT_REAL_OUTPUT);
    int cx = filter_image.cols/2;
    int cy = filter_image.rows/2;
    Mat q0(filter_image, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(filter_image, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(filter_image, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(filter_image, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(filter_image, filter_image, 0, 255, NORM_MINMAX);
    // could resize here
    flip(filter_image, filter_image, 0);
    imshow("filter image", filter_image);

    imshow("input image", this->curr_image.frame);
    waitKey(10);
    /****** end show images *********/
#endif

    float PSR_val = calcPSR(idft_correlation);
    DLOG(INFO) << "PSR = " << PSR_val;

    Point2i maxLoc;
    if(PSR_val >= this->_PSR_ratio[1]){
        minMaxLoc(idft_correlation, nullptr, nullptr, nullptr, &maxLoc);
        Mat new_output = Mat::zeros(mat_correlation.size(), CV_32F);
        maskDesiredG(new_output, maxLoc.x, maxLoc.y);
        new_output = calcDFT(new_output, false);
        new_output.copyTo(this->curr_image.filter_output);
    } else if(PSR_val > this->_PSR_ratio[0]){
        maxLoc.x = -1;
        maxLoc.y = -1;
    } else{
        maxLoc.x = -2;
        maxLoc.y = -2;
    }

    return maxLoc;
}

float ly::Tracker::calcPSR(const Mat &correlation_mat) {

    double max_val = 0;
    Point max_loc;
    Mat PSR_mask = Mat::ones(correlation_mat.rows,correlation_mat.cols, CV_8U);
    Scalar mean,stddev;

    minMaxLoc(correlation_mat,NULL,&max_val,NULL,&max_loc);     //Get location of max arg

    //Define PSR mask
    int win_size = floor(this->_PSR_mask/2);
    Rect mini_roi = Rect(std::max(max_loc.x - win_size,0), std::max(max_loc.y - win_size,0), this->_PSR_mask, this->_PSR_mask);

    //Handle image boundaries
    if ( (mini_roi.x+mini_roi.width) > PSR_mask.cols )
    {
        mini_roi.width = PSR_mask.cols - mini_roi.x;
    }
    if ( (mini_roi.y+mini_roi.height) > PSR_mask.rows )
    {
        mini_roi.height = PSR_mask.rows - mini_roi.y;
    }

    Mat temp = PSR_mask(mini_roi);
    temp *= 0;
    meanStdDev(correlation_mat,mean,stddev,PSR_mask);   //Compute matrix mean and std

    return (max_val - mean.val[0]) / stddev.val[0];     //Compute PSR
}

void ly::Tracker::update(Point2i new_location) {

    updateFilter();
    this->prev_image = this->curr_image;
    updateRoi(new_location, false);
}

void ly::Tracker::updateFilter() {

    Mat Ai,Bi,Ai_1,Bi_1,A,B,filter,eps,eps_1;
    mulSpectrums(this->curr_image.filter_output, this->curr_image.image_sepctrum,
                 Ai, 0, true);
    mulSpectrums(this->prev_image.filter_output, this->prev_image.image_sepctrum,
                 Ai_1, 0, true);
    mulSpectrums(this->curr_image.image_sepctrum, this->curr_image.image_sepctrum,
                 Bi, 0, true);
    mulSpectrums(this->prev_image.image_sepctrum, this->prev_image.image_sepctrum,
                 Bi_1, 0, true);

    if(this->_eps){
        eps = createEps(Bi);
        Bi += eps;

        eps_1 = createEps(Bi_1);
        Bi_1 += eps;
    }

    A = ((1.0-this->_learning_rate)*Ai) + (this->_learning_rate*Ai_1);
    B = ((1.0-this->_learning_rate)*Bi) + (this->_learning_rate*Bi_1);
    dftDiv(A, B, filter);
    filter.copyTo(this->_filter);

}

void ly::Tracker::updateRoi(Point2i new_center, bool scale_rot) {

    int diff_x, diff_y;
    this->prev_roi = this->curr_roi;

    this->curr_roi.roi_center = new_center;
    new_center.x += prev_roi.roi.x;
    new_center.y += prev_roi.roi.y;

    diff_x = new_center.x - round(this->curr_roi.roi.width/2);
    diff_y = new_center.y - round(this->curr_roi.roi.height/2);

    if (diff_x < 0)
    {
        this->curr_roi.roi.x = 0;
    }
    else if( (diff_x + this->curr_roi.roi.width) >= this->_image_size.width )
    {
        this->curr_roi.roi.x = this->_image_size.width - this->curr_roi.roi.width -1;
    }else{
        this->curr_roi.roi.x = diff_x;
    }

    if (diff_y < 0)
    {
        this->curr_roi.roi.y = 0;
    }
    else if( (diff_y + this->curr_roi.roi.height) >= this->_image_size.height )
    {
        this->curr_roi.roi.y = this->_image_size.height - this->curr_roi.roi.height -1;
    }else{
        this->curr_roi.roi.y = diff_y;
    }

    this->curr_roi.roi.width = this->prev_roi.roi.width;
    this->curr_roi.roi.height = this->prev_roi.roi.height;
}

bool ly::Tracker::isInit() const {
    return _init;
}
