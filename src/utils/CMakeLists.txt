#find_package(glog REQUIRED)

set(CMAKE_PREFIX_PATH "/usr/local/include/opencv4/")
set(OpenCV_DIR /usr/local/include/opencv4/)
find_package(OpenCV 4 REQUIRED)
find_package(glog REQUIRED)
# find_package(OpenCV REQUIRED)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
	    /usr/local/include/glog
)

add_library(utils
        src/Config.cpp
        src/Log.cpp
        src/Params.cpp
        src/Thread.cpp
        )

target_link_libraries(utils
        #-lglog
	glog::glog
        -ljsoncpp
        ${OpenCV_LIBS}
        )

# 将model文件复制到build目录下

file(COPY model DESTINATION ${CMAKE_BINARY_DIR})