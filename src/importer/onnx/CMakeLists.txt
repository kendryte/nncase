include(TestBigEndian)

set(ONNX_SOURCE_DIR ${THIRD_PARTY}/onnx)

function(onnx_gen_proto PROTO_FILE PROTO_IN_FILE)
    set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    set(${PROTO_FILE})

    set(ONNX_NAMESPACE "onnx")

    if("${PYTHON_EXECUTABLE}" STREQUAL "")
        set(_python_exe "python")
    else()
        set(_python_exe "${PYTHON_EXECUTABLE}")
    endif()

    set(GEN_PROTO_PY ${ONNX_SOURCE_DIR}/onnx/gen_proto.py)

    get_filename_component(PROTO_FILE_WE ${PROTO_IN_FILE} NAME_WE)

    set(GENERATED_PROTO "${PROTO_FILE_WE}.proto")
    message("GG: ${GENERATED_PROTO}")

    set(GEN_PROTO_ARGS
        -p
        "${ONNX_NAMESPACE}"
        -o
        "${OUTPUT_DIR}"
        "${PROTO_FILE_WE}"
        )

    add_custom_command(OUTPUT "${GENERATED_PROTO}"
        COMMAND "${_python_exe}" "${GEN_PROTO_PY}"
        ARGS ${GEN_PROTO_ARGS}
        DEPENDS ${PROTO_IN_FILE}
        COMMENT "Running gen_proto.py on ${PROTO_IN_FILE}"
        VERBATIM)
    message("Generated: ${GENERATED_PROTO}")
    set(${PROTO_FILE} ${GENERATED_PROTO} PARENT_SCOPE)

    set_source_files_properties(${GENERATED_PROTO} PROPERTIES GENERATED TRUE)
endfunction()

function(onnx_proto_generate_cpp SRC HDR PROTO_FILE)
    set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/onnx)

    get_filename_component(PARENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} DIRECTORY)

    set(${SRC})
    set(${HDR})

    set(ONNX_PROTOC_ARGS
        onnx/${PROTO_FILE}
        -I
        ${PARENT_BINARY_DIR}
        --cpp_out
        ${CMAKE_CURRENT_BINARY_DIR})

    get_filename_component(PROTO_FILE_WE ${PROTO_FILE} NAME_WE)

    set(OUTPUT_SRC "${OUTPUT_DIR}/${PROTO_FILE_WE}.pb.cc")
    set(OUTPUT_HDR "${OUTPUT_DIR}/${PROTO_FILE_WE}.pb.h")

    message("OUTPUT_SRC: ${OUTPUT_SRC}")
    message("OUTPUT_HDR: ${OUTPUT_HDR}")

    add_custom_command(
        OUTPUT "${OUTPUT_SRC}" "${OUTPUT_HDR}"
        COMMAND ${Protobuf_PROTOC_EXECUTABLE} ARGS ${ONNX_PROTOC_ARGS}
        DEPENDS ${PROTO_FILE}
        WORKING_DIRECTORY ${PARENT_BINARY_DIR}
        COMMENT "Running C++ protocol buffer compiler on ${PROTO_FILE}"
        VERBATIM)

    set(${SRC} ${OUTPUT_SRC} PARENT_SCOPE)
    set(${HDR} ${OUTPUT_HDR} PARENT_SCOPE)

    set_source_files_properties(${OUTPUT_SRC} ${OUTPUT_HDR} PROPERTIES GENERATED TRUE)
endfunction()

onnx_gen_proto(ONNX_PROTO ${ONNX_SOURCE_DIR}/onnx/onnx.in.proto)
onnx_proto_generate_cpp(ONNX_PROTO_SOURCE ONNX_PROTO_HEADER ${ONNX_PROTO})

onnx_gen_proto(ONNX_OPS_PROTO ${ONNX_SOURCE_DIR}/onnx/onnx-operators.in.proto)
onnx_proto_generate_cpp(ONNX_OPS_PROTO_SOURCE ONNX_OPS_PROTO_HEADER ${ONNX_OPS_PROTO})

add_library(onnx_proto ${ONNX_PROTO_SOURCE} ${ONNX_OPS_PROTO_SOURCE})
target_include_directories(onnx_proto PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(onnx_proto PRIVATE CONAN_PKG::protobuf)

test_big_endian(BIG_ENDIAN)

set(ONNX_IMPORTER_SOURCES
    onnx_importer.cpp
    )

set(ONNX_IMPORTER_OPS_SOURCES
    ops/activations.cpp
    ops/binary.cpp
    ops/concat.cpp
    ops/constant.cpp
    ops/conv.cpp
    ops/lp_normalization.cpp
    ops/matmul.cpp
    ops/pad.cpp
    ops/pool.cpp
    ops/reduce.cpp
    ops/reshape.cpp
    ops/resize.cpp
    ops/slice.cpp
    ops/softmax.cpp
    ops/squeeze.cpp
    ops/transpose.cpp
    ops/unary.cpp
    )

add_library(onnx_importer ${ONNX_IMPORTER_SOURCES} ${ONNX_IMPORTER_OPS_SOURCES})
target_compile_definitions(onnx_importer PRIVATE NATIVE_IS_BIG_ENDIAN=${BIG_ENDIAN})
target_include_directories(onnx_importer PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
get_filename_component(PARENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
target_include_directories(onnx_importer PUBLIC ${PARENT_SOURCE_DIR}/include)
target_link_libraries(onnx_importer PUBLIC hlir CONAN_PKG::boost)
target_link_libraries(onnx_importer PRIVATE onnx_proto CONAN_PKG::protobuf)
