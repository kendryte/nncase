 include(TestBigEndian)
test_big_endian(BIG_ENDIAN)

set(PNNX_IMPORTER_SOURCES
    pnnx_importer.cpp
    ir.cpp
    storezip.cpp
    )

set(PNNX_IMPORTER_OPS_SOURCES
    ops/input.cpp
    ops/output.cpp
    ops/relu.cpp
    ops/relu6.cpp
    ops/conv2d.cpp
    )

add_library(pnnx_importer ${PNNX_IMPORTER_SOURCES} ${PNNX_IMPORTER_OPS_SOURCES})

target_compile_definitions(pnnx_importer PRIVATE NATIVE_IS_BIG_ENDIAN=${BIG_ENDIAN})
target_include_directories(pnnx_importer PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
get_filename_component(PARENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
target_include_directories(pnnx_importer PUBLIC ${PARENT_SOURCE_DIR}/include)

target_link_libraries(pnnx_importer PUBLIC ir)
set_target_properties(pnnx_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
