//
// Created by zhiyu on 2021/8/20.
//

#ifndef AUTOAIM_CONFIG_H
#define AUTOAIM_CONFIG_H

#include <string>
#include <fstream>
#include "glog/logging.h"
#include "jsoncpp/json/json.h"

#include "Params.h"
#include "../../driver/include/VideoCapture.h"

using namespace std;

namespace ly{
    /**
     * @brief: 解析配置文件
     */
    class Config { 
    public:
        Config()  = default;
        explicit Config(const string& path);
        void parse();
        void chooseCameraType(VideoCapture*& video);
    private:
        string json_file_path;
    };
}

#endif //AUTOAIM_CONFIG_H
