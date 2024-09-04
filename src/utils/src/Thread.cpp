//
// Created by zhiyu on 2021/8/21.
//

#include "Thread.h"

namespace ly{
    bool Thread::image_is_update = false;

    condition_variable Thread::cond_is_update;
    condition_variable Thread::cond_is_process;

    // mutex Thread::mtx;
    mutex Thread::mtx_image;    // 是一个所
    mutex Thread::mtx_video;
}