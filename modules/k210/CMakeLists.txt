cmake_minimum_required (VERSION 3.8)

include_directories(include)
add_subdirectory(src/kernels)
add_subdirectory(src/runtime)

if (BUILDING_RUNTIME)
    if (ENABLE_K210_RUNTIME)
        set(SRCS src/dummy.cpp)
        
        add_library(nncase_rt_modules_k210 STATIC ${SRCS})
        target_include_directories(nncase_rt_modules_k210 PRIVATE include)
        target_link_libraries(nncase_rt_modules_k210 PRIVATE runtime_k210 kernels_k210)
        set_target_properties(nncase_rt_modules_k210 PROPERTIES
                                                  OUTPUT_NAME "nncase.rt_modules.k210")
        
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

        install(TARGETS nncase_rt_modules_k210 EXPORT nncaseruntimeTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION include
        )
        
        configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/nncase_rt_modules_k210Config.cmake.in nncase_rt_modules_k210Config.cmake @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncase_rt_modules_k210Config.cmake DESTINATION lib/cmake/nncaseruntime)
    endif()
else()
    add_subdirectory(src/codegen)
    add_subdirectory(src/ir)
    add_subdirectory(src/schedule)
    add_subdirectory(src/transforms/k210)
    add_subdirectory(src/evaluator)
    
    set(SRCS src/dummy.cpp)
    
    add_library(nncase_modules_k210 SHARED ${SRCS})
    target_include_directories(nncase_modules_k210 PUBLIC include)
    target_link_libraries(nncase_modules_k210 PRIVATE codegen_k210 ir_k210 evaluator_k210 schedule_k210 transforms_k210 simulator_k210 kernels_k210 nncase)
    set_target_properties(nncase_modules_k210 PROPERTIES
                                              OUTPUT_NAME "nncase.modules.k210")
    install(TARGETS nncase_modules_k210
            COMPONENT nncase-runtime)
endif()
