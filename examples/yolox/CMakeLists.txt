cmake_minimum_required(VERSION 3.19)
project(test_decoder)
set(CMAKE_CXX_STANDARD 14)

include_directories(k210/yolox_detect_example)

add_executable(test_decoder 
    cpu/main.c
    k210/yolox_detect_example/yolox.cpp
    )