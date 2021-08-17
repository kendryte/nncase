include(TestBigEndian)
test_big_endian(BIG_ENDIAN)

set(NCNN_IMPORTER_SOURCES
    ncnn_importer.cpp
    datareader.cpp
    mat.cpp
    modelbin.cpp
    paramdict.cpp
    )

set(NCNN_IMPORTER_OPS_SOURCES
    ops/activations.cpp
    ops/convolution.cpp
    )

add_library(ncnn_importer ${NCNN_IMPORTER_SOURCES} ${NCNN_IMPORTER_OPS_SOURCES})

target_compile_definitions(ncnn_importer PRIVATE NATIVE_IS_BIG_ENDIAN=${BIG_ENDIAN})
target_include_directories(ncnn_importer PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
get_filename_component(PARENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
target_include_directories(ncnn_importer PUBLIC ${PARENT_SOURCE_DIR}/include)

target_link_libraries(ncnn_importer PUBLIC ir)
set_target_properties(ncnn_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
