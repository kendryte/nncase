﻿cmake_minimum_required (VERSION 3.13)

target_sources(ir PRIVATE
    call.cpp
    copy.cpp
    conv2d.cpp
    conv2d_transpose.cpp
    convert.cpp
    fused_unary.cpp
    matmul.cpp
    transpose.cpp
    reduce.cpp
    reduce_window2d.cpp
    binary.cpp
    concat.cpp
    clamp.cpp
    constant.cpp
    quantize.cpp
    dequantize.cpp
    unary.cpp
    pad.cpp
    bitcast.cpp
    resize_image.cpp
    slice.cpp
    table_lookup.cpp
    broadcast.cpp
    take.cpp
    space_to_batch.cpp
    batch_to_space.cpp
    split.cpp
    gather.cpp
    gather_nd.cpp
    onehot.cpp)
