cmake_minimum_required(VERSION 3.0)
project(HelloInsight)

# add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 11)

# Set(Qt5_DIR /opt/homebrew/Cellar/qt@5/5.15.2/lib/cmake/Qt5)
SET(ITK_DIR "/home/kuka/SVR/ITK/build")
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

include_directories( /home/kuka/SVR/program/include/optim )
include_directories( /home/kuka/SVR/program/include/itk_img )
include_directories( /home/kuka/SVR/program/include/optimization )
include_directories( "/home/kuka/SVR/eigen" )
include_directories( /home/kuka/SVR/program/include/elastix )

# FIND_PACKAGE( OpenMP REQUIRED)
# if(OPENMP_FOUND)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

# find_package( OpenCL REQUIRED )
# include_directories( ${OPENCL_INCLUDE_DIRS} )
# message(OPENCL_FOUND)

# add_executable(Test test.cpp )
# target_link_libraries(Test ${ITK_LIBRARIES})

add_executable(HelloInsight HelloInsight.cpp ITKImgProcess.cpp Optimization.cpp)
target_link_libraries(HelloInsight ${ITK_LIBRARIES})

