# 🛠️ INSTALL_WSL.md

## 一、🎯 OpenVINO 安装
按照 [https://docs.openvino.ai/2024/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_4_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT](https://docs.openvino.ai/2024/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_4_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT) 进行安装。

## 二、🎯 OpenCV 安装
按照 [https://opencv.org/get-started/](https://opencv.org/get-started/) 进行安装。

## 三、🎯 gtest & boost 安装
使用命令 `apt install libgtest-dev libboost-all-dev` 进行安装。

## 四、🎯 glog 安装
1. `git clone https://github.com/google/glog`
2. `cd glog/ && mkdir build && cd build && cmake.. && make && make test`
3. `make install && cd../..`

## 五、🎯 eigen3 安装
1. `wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz && tar zxf eigen-3.3.9.tar.gz`
2. `cd eigen-3.3.9/ && mkdir build && cd build && cmake.. && make && make install && cd../..`

## 六、🎯 Sophus 安装
1. `git clone https://github.com/strasdat/Sophus.git`
2. `cd Sophus/ && git checkout a621ff`
3. 打开 `sophus/so2.cpp` 文件进行如下修改：
   - 第 32 行第 21 处 `"unit_complex_.real() = 1.;"` 修改为 `"unit_complex_.real(1.);"`
   - 第 33 行第 21 处 `"unit_complex_.imag() = 0.;"` 修改为 `"unit_complex_.real(0.);"`
4. `cmake. && make && make install && cd..`

## 七、🎯 CeresSolver 安装
1. `wget http://ceres-solver.org/ceres-solver-2.2.0.tar.gz && tar zxf ceres-solver-2.2.0.tar.gz`
2. `cd ceres-solver-2.2.0/ && mkdir build && cd build && cmake.. && make && make test`
3. `make install && cd../..`

## 八、🎯 jsoncpp 安装
1. `git clone https://github.com/open-source-parsers/jsoncpp`
2. `cd jsoncpp/ && mkdir build && cd build && cmake.. && make && make test`
3. `make install && cd../..`

## 九、🎯 Galaxy 安装
1. `wget https://gb.daheng-imaging.com/CN/Software/Cameras/Linux/Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221.zip`
2. `unzip Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221.zip && cd Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221`
3. `bash *.run`
4. `cd..`

## 十、🎯 G2O 安装
`git clone https://github.com/RainerKuemmerle/g2o`

## 十一、🎯 HUST_HeroAim_2024 安装
1. 进入 `HUST_HeroAim_2024` 目录，打开 `./CMakeLists.txt` 文件进行如下修改：
   - 第 6 行添加 `"set(Sophus_DIR /path/to/source/code/Sophus)"`
   - 第 6 行添加 `"list(APPEND CMAKE_MODULE_PATH /path/to/source/code/g2o/cmake_modules/)"`
2. 打开 `./src/utils/include/Config.h` 文件，将第 11 行第 11 处 `"jsoncpp/json/json.h"` 修改为 `"json/json.h"`