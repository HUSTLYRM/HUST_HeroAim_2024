# find_package(Pangolin REQUIRED)
find_package(Ceres REQUIRED)
LIST(APPEND CMAKE_MODULE_PATH /home/nvidia/g2o/cmake_modules)
cmake_policy(SET CMP0074 NEW)

SET(G2O_ROOT /usr/local/include/g2o)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

# find_package(CSparse REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

include_directories(
        include
        ../utils/include
        # 去除冲突
        # /usr/include/eigen3
        /usr/local/include/eigen3
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        )

add_library(pose_estimate
        src/PoseSolver.cpp
        src/Predictor_main.cpp
        src/Kalman.cpp
        src/EulerAngleKF.cpp
        src/NormalEKF.cpp
        )

target_link_libraries(pose_estimate
        utils

        /usr/local/lib/libSophus.so
        ${CERES_LIBRARIES}
        #${G2O_LIBS}
        )
