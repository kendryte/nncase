cmake_minimum_required (VERSION 3.8)

set(SRCS compiler.cpp
         simulator.cpp)

add_library(nncase SHARED ${SRCS})
target_link_libraries(nncase PRIVATE data ir tflite_importer kernels evaluator importer schedule codegen codegen_stackvm transforms targets simulator simulator_stackvm plugin)
target_link_libraries(nncase PUBLIC gsl::gsl-lite xtensor::xtensor mpark_variant::mpark_variant)
target_compile_definitions(nncase PRIVATE -DNNCASE_DLL)
target_compile_definitions(nncase PUBLIC -DNNCASE_SHARED_LIBS)
target_link_libraries(nncase PRIVATE magic_enum::magic_enum fmt::fmt)

install(TARGETS nncase EXPORT nncaseTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

install(EXPORT nncaseTargets
        DESTINATION lib/cmake/nncase)

configure_file(${CMAKE_CURRENT_LIST_DIR}/../../cmake/nncaseConfig.cmake.in nncaseConfig.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncaseConfig.cmake DESTINATION lib/cmake/nncase)
