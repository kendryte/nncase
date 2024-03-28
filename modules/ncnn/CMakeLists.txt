cmake_minimum_required (VERSION 3.8)

if (BUILDING_RUNTIME)
    add_definitions(-DBUILDING_RUNTIME)
endif()

include_directories(include)
add_subdirectory(src/runtime)

if (BUILDING_RUNTIME)
    add_library(nncase_rt_modules_ncnn STATIC ${SRCS})
    target_include_directories(nncase_rt_modules_ncnn PRIVATE include)
    target_link_libraries(nncase_rt_modules_ncnn PRIVATE runtime_ncnn)
    set_target_properties(nncase_rt_modules_ncnn PROPERTIES
            OUTPUT_NAME "nncase.rt_modules.ncnn")

    install(DIRECTORY include/nncase/kernels
            DESTINATION include/nncase
            COMPONENT nncase-headers
            FILES_MATCHING
            PATTERN "*.def"
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN "*.td"
            PATTERN "*.inc"
            PATTERN "LICENSE.TXT"
            )

    install(DIRECTORY include/nncase/runtime
            DESTINATION include/nncase
            COMPONENT nncase-headers
            FILES_MATCHING
            PATTERN "*.def"
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN "*.td"
            PATTERN "*.inc"
            PATTERN "LICENSE.TXT"
            )

    install(TARGETS nncase_rt_modules_ncnn EXPORT nncase_rt_modules_ncnnTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION include
            )

    install(EXPORT nncase_rt_modules_ncnnTargets
            DESTINATION lib/cmake/nncase_rt_modules_ncnn)
else()
    add_library(nncase_modules_ncnn SHARED ${SRCS})
    target_include_directories(nncase_modules_ncnn PUBLIC include)
    target_link_libraries(nncase_modules_ncnn PRIVATE simulator_ncnn)
    set_target_properties(nncase_modules_ncnn PROPERTIES
            OUTPUT_NAME "nncase.simulator.ncnn")
    install(TARGETS nncase_modules_ncnn
            COMPONENT nncase-runtime)

endif()
