cmake_minimum_required (VERSION 3.8)

include_directories(include)
#add_subdirectory(src/kernels)
add_subdirectory(src/runtime)

if (BUILDING_RUNTIME)
    if (ENABLE_VULKAN_RUNTIME)
        add_library(nncase_rt_modules_cpu STATIC ${SRCS})
        target_include_directories(nncase_rt_modules_cpu PRIVATE include)
        target_link_libraries(nncase_rt_modules_cpu PRIVATE runtime_cpu)
        set_target_properties(nncase_rt_modules_cpu PROPERTIES
                                                       OUTPUT_NAME "nncase.rt_modules.cpu")

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

        install(TARGETS nncase_rt_modules_cpu EXPORT nncaseruntimeTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION include
        )

        configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/nncase_rt_modules_cpuConfig.cmake.in nncase_rt_modules_cpuConfig.cmake @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncase_rt_modules_cpuConfig.cmake DESTINATION lib/cmake/nncaseruntime)
    endif()
else()
    add_library(nncase_modules_cpu SHARED ${SRCS})
    target_include_directories(nncase_modules_cpu PUBLIC include)
    target_link_libraries(nncase_modules_cpu PRIVATE simulator_cpu nncasebase)
    set_target_properties(nncase_modules_cpu PROPERTIES OUTPUT_NAME "nncase.modules.cpu")
    install(TARGETS nncase_modules_cpu
            COMPONENT nncase-runtime)
endif()
