cmake_minimum_required (VERSION 3.13)

target_sources(ir PRIVATE
    broadcast.cpp
    concat.cpp
    cast.cpp
    copy.cpp
    gather.cpp
    gather_nd.cpp
    reshape.cpp
    slice.cpp
    split.cpp
    transpose.cpp
    )
