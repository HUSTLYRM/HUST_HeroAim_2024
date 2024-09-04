//
// Created by zhiyu on 2021/8/20.
//

#ifndef AUTOAIM_LOG_H
#define AUTOAIM_LOG_H

#include "glog/logging.h"

#define DEBUG_MODE(op) (DLOG_IF(INFO, (op, false)))

namespace ly
{
    class Log
    {
    public:
        Log() = default;
        ~Log();
        explicit Log(const char *argv0);
        void init(const char *argv0);
    };
} 

#endif //AUTOAIM_LOG_H
