cmake_minimum_required (VERSION 3.8)

set(SRCS dataset.cpp)

add_library(data OBJECT ${SRCS})
target_link_libraries(data PUBLIC xtensor::xtensor gsl::gsl-lite)
target_compile_definitions(data PUBLIC -DNNCASE_DLL)
target_link_libraries(data PRIVATE opencv::core opencv::imgproc opencv::imgcodecs)
set_property(TARGET data PROPERTY POSITION_INDEPENDENT_CODE ON)