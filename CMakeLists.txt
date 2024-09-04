cmake_minimum_required(VERSION 3.5)
project(AutoAim)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "/usr/local/include/opencv4/")
set(OpenCV_DIR /usr/local/include/opencv4/)
find_package(OpenCV 4 REQUIRED)

# set(InferenceEngine_DIR /opt/intel/openvino_2022/runtime/cmake)
# set(ngraph_DIR /opt/intel/openvino_2022/runtime/cmake)
# find_package(InferenceEngine REQUIRED)
# find_package(ngraph REQUIRED)
# set(BOOST_ROOT "/usr/include/")
find_package(Boost REQUIRED COMPONENTS system filesystem)

add_subdirectory(src/armor_detector)
add_subdirectory(src/pose_estimate)
add_subdirectory(src/driver)
add_subdirectory(src/utils)

include_directories(
        src/
        src/utils/include
        /usr/local/include/eigen3
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
#         ${CMAKE_CURRENT_SOURCE_DIR}
#         ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(AutoAim
        src/AutoAim.cpp
)

add_executable(train
       src/trainSVM.cpp
)

add_executable(testArray
       src/utils/testArray.cpp
)


target_link_libraries(AutoAim
        utils
        driver
        armor_detector
        ${OpenCV_LIBS}
        -lboost_system
        -lboost_filesystem
        -lpthread
)

target_link_libraries(train
       utils
       driver
       armor_detector
       ${Boost_LIBRARIES}
#        ${InferenceEngine_LIBRARIES}
       ${OpenCV_LIBS}
)

target_link_libraries(testArray
       utils
       ${OpenCV_LIBS}
)