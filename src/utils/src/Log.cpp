//
// Created by zhiyu on 2021/8/20.
//

#include "Log.h"

using namespace ly;
using namespace google;

Log::Log(const char* argv0) {
    FLAGS_logtostderr = false;
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_logbufsecs = 0;

    InitGoogleLogging(argv0);
    SetLogDestination(INFO, "../logs/INFO_");
    SetLogDestination(WARNING, "../logs/WARNING_");
    SetLogDestination(ERROR, "../logs/ERROR_");
    SetLogDestination(FATAL, "../logs_FATAL");
    SetLogFilenameExtension("log");

}

void Log::init(const char *argv0) {
    FLAGS_logtostderr = false;
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_logbufsecs = 0;

    InitGoogleLogging(argv0);
    SetLogDestination(INFO, "../logs/INFO_");
    SetLogDestination(WARNING, "../logs/WARNING_");
    SetLogDestination(ERROR, "../logs/ERROR_");
    SetLogDestination(FATAL, "../logs_FATAL");
    SetLogFilenameExtension("log");
}

Log::~Log() {
    ShutdownGoogleLogging();
}
