#include "AutoAim.h"

using namespace ly;

int main(int argc, char** argv){
    
    Params_ToSerialPort params_to_serial_port(&SerialParam::recv_data);
    Params_ToVideo params_to_video;
    Params_ToDetector params_to_detector;

    /******* init log ********/
    auto log = new Log();
    log->init(argv[0]);

    /******* init config ********/
    auto config = new Config(confog_file_path);
    config->parse();

    /******* init serial port read ******/
    auto serial_port = new SerialPort(SerialParam::device_name);
    thread serial_port_thread(&SerialPort::read_data, serial_port, ref(params_to_serial_port));

    // // auto second = std::chrono::steady_clock::now();
    // while(true){
    //     // double delta_t = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - second).count()/1000.0;
    //     // if(delta_t < 1000){
    //     //     SerialParam::send_data.time_stamp = 0;
    //     //     serial_port->writeData(&SerialParam::send_data);
    //     //     continue;
    //     // }
    //     // else second = std::chrono::steady_clock::now();
    //     // auto zero = std::chrono::steady_clock::now();
    //     SerialParam::send_data.time_stamp = 1;
    //     SerialParam::send_data.yaw = 100;
    //     // DLOG(INFO) << "send: " << delta_t;
    //     serial_port->writeData(&SerialParam::send_data);
    //     // while(SerialParam::recv_data.time_stamp == 0);
    //     // DLOG(INFO) << SerialParam::recv_data.time_stamp;
    //     // if(SerialParam::recv_data.time_stamp) 
    //     // DLOG(INFO) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - zero).count()/1000.0;
    // }

    /******* init camera ********/
    ly::VideoCapture* video;
    video->chooseCameraType(video);
    thread video_thread(&ly::VideoCapture::startCapture, video, ref(params_to_video));

    /******* init detector ********/
    auto detector = new Detector();
    detector->setParams(params_to_video, params_to_serial_port);
    thread detector_thread(&Detector::startDetect, detector, ref(params_to_detector), serial_port);


   /******** init video writer ****/
   if(GlobalParam::SAVE_VIDEO){
        auto saver = new VideoSaver();
        thread saver_thread(&VideoSaver::SaveVideo, saver, params_to_video.frame_pp);
        saver_thread.join();
   }

    serial_port_thread.join();
    video_thread.join();
    detector_thread.join();
    return 0;
}
