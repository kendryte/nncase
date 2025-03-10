cmake_minimum_required (VERSION 3.8)

include_directories(include)
#add_subdirectory(src/kernels)
add_subdirectory(src/runtime)

# if (BUILDING_RUNTIME)
#     # if (ENABLE_VULKAN_RUNTIME)
#     #     set(SRCS src/dllmain.cpp)

#     #     add_library(nncase_rt_modules_vulkan STATIC ${SRCS})
#     #     target_include_directories(nncase_rt_modules_vulkan PRIVATE include)
#     #     target_link_libraries(nncase_rt_modules_vulkan PRIVATE runtime_vulkan kernels_vulkan)
#     #     set_target_properties(nncase_rt_modules_vulkan PROPERTIES
#     #                                                    OUTPUT_NAME "nncase.rt_modules.vulkan")

#     #     install(DIRECTORY include/nncase/kernels
#     #         DESTINATION include/nncase
#     #         COMPONENT nncase-headers
#     #         FILES_MATCHING
#     #         PATTERN "*.def"
#     #         PATTERN "*.h"
#     #         PATTERN "*.hpp"
#     #         PATTERN "*.td"
#     #         PATTERN "*.inc"
#     #         PATTERN "LICENSE.TXT"
#     #         )

#     #     install(DIRECTORY include/nncase/runtime
#     #         DESTINATION include/nncase
#     #         COMPONENT nncase-headers
#     #         FILES_MATCHING
#     #         PATTERN "*.def"
#     #         PATTERN "*.h"
#     #         PATTERN "*.hpp"
#     #         PATTERN "*.td"
#     #         PATTERN "*.inc"
#     #         PATTERN "LICENSE.TXT"
#     #         )

#     #     install(TARGETS nncase_rt_modules_vulkan EXPORT nncaseruntimeTargets
#     #         ARCHIVE DESTINATION lib
#     #         LIBRARY DESTINATION lib
#     #         RUNTIME DESTINATION bin
#     #         INCLUDES DESTINATION include
#     #     )

#     #     configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/nncase_rt_modules_vulkanConfig.cmake.in nncase_rt_modules_vulkanConfig.cmake @ONLY)
#     #     install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncase_rt_modules_vulkanConfig.cmake DESTINATION lib/cmake/nncaseruntime)
#     # endif()
# else()
#     add_subdirectory(src/codegen)
#     add_subdirectory(src/transforms/vulkan)
    
#     set(SRCS src/dllmain.cpp)
    
#     add_library(nncase_modules_vulkan SHARED ${SRCS})
#     target_include_directories(nncase_modules_vulkan PUBLIC include)
#     target_link_libraries(nncase_modules_vulkan PRIVATE codegen_vulkan transforms_vulkan simulator_vulkan nncase)
#     set_target_properties(nncase_modules_vulkan PROPERTIES
#                                                 OUTPUT_NAME "nncase.modules.vulkan")
#     if(MSVC)
#         target_link_libraries(nncase_modules_vulkan PRIVATE vulkan_templates_rc)
#     endif()
#     install(TARGETS nncase_modules_vulkan
#             COMPONENT nncase-runtime)
# endif()
