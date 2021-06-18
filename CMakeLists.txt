cmake_minimum_required(VERSION 3.0)
project(HelloInsight)

# add_compile_options(-std=c++17)

Set(Qt5_DIR /opt/homebrew/Cellar/qt@5/5.15.2/lib/cmake/Qt5)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
include_directories( include/optim )
include_directories( include/itk_img )
include_directories( include/optimization )
include_directories( "/usr/local/include/eigen3" )

add_executable(Test test.cpp )
target_link_libraries(Test ${ITK_LIBRARIES})

add_executable(HelloInsight HelloInsight.cpp ITKImgProcess.cpp Optimization.cpp)
target_link_libraries(HelloInsight ${ITK_LIBRARIES})

