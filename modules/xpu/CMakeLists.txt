cmake_minimum_required (VERSION 3.8)

include_directories(include)
#add_subdirectory(src/kernels)
add_subdirectory(src/runtime)

if (BUILDING_RUNTIME)
    if (ENABLE_XPU_RUNTIME)
        add_library(nncase_rt_modules_xpu STATIC ${SRCS})
        target_include_directories(nncase_rt_modules_xpu PRIVATE include)
        target_link_libraries(nncase_rt_modules_xpu PRIVATE runtime_xpu)
        set_target_properties(nncase_rt_modules_xpu PROPERTIES
                                                       OUTPUT_NAME "nncase.rt_modules.xpu")

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

        install(TARGETS nncase_rt_modules_xpu EXPORT nncaseruntimeTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION include
        )

        configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/nncase_rt_modules_xpuConfig.cmake.in nncase_rt_modules_xpuConfig.cmake @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncase_rt_modules_xpuConfig.cmake DESTINATION lib/cmake/nncaseruntime)

        if (K230_LINUX_SDK_DIR)
            add_subdirectory(examples)
        endif()
    endif()
else()
    add_library(nncase_modules_xpu SHARED ${SRCS})
    target_include_directories(nncase_modules_xpu PUBLIC include)
    target_link_libraries(nncase_modules_xpu PRIVATE simulator_xpu nncasebase)
    set_target_properties(nncase_modules_xpu PROPERTIES OUTPUT_NAME "nncase.simulator.xpu")
    install(TARGETS nncase_modules_xpu
            COMPONENT nncase-runtime)
endif()
