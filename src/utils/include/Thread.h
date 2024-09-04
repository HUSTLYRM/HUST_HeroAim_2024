//
// Created by zhiyu on 2021/8/21.
//

#ifndef AUTOAIM_THREAD_H
#define AUTOAIM_THREAD_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace std;

namespace ly
{

    class Thread
    {
    public:
        // 表示图片是否已经更新
        static bool image_is_update;

        // 条件变量
        // 用于在 Detector 和 VideoCapture 之间控制同步关系, 明白即可
        static condition_variable cond_is_update;
        static condition_variable cond_is_process;

        // 静态互斥锁
        // static mutex mtx;
        static mutex mtx_image;     // 相机读取 以及 detector的互斥锁
        static mutex mtx_video;     // 保存视频 以及 detector的互斥锁
    };
}

#endif //AUTOAIM_THREAD_H
