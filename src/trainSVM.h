//
// Created by zhiyu on 2021/9/11.
//

#ifndef AUTOAIM_TRAINSVM_H
#define AUTOAIM_TRAINSVM_H

#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
//#undef BOOST_NO_CXX11_SCOPED_ENUMS

#include "armor_detector/include/Score.h"

namespace ly{
    string picRoot = "../src/utils/data/pictures/armors";
}

#endif //AUTOAIM_TRAINSVM_H
