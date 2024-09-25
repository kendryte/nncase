# cmake_minimum_required (VERSION 3.13)

# set(SRCS runtime_module.cpp
#          runtime_function.cpp
#          vulkan_error.cpp
#          vulkan_context.cpp
#          op_reader.cpp
#          ops/ldbuf.cpp
#          ops/ldbufbarrier.cpp
#          ops/ldbufcopy.cpp
#          ops/copybuf.cpp
#          ops/ldpipeline.cpp
#          ops/dispatch.cpp
#          ops/barrier.cpp)

# if (BUILDING_RUNTIME)
#     if (ENABLE_VULKAN_RUNTIME)
#         add_library(runtime_vulkan OBJECT ${SRCS})
#         target_compile_definitions(runtime_vulkan PRIVATE -DVULKAN_HPP_NO_EXCEPTIONS)
#         target_link_libraries(runtime_vulkan PUBLIC nncaseruntime)
#         target_link_libraries(runtime_vulkan PRIVATE Vulkan::Vulkan)
#         set_target_properties(runtime_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
#         install(TARGETS runtime_vulkan EXPORT nncaseruntimeTargets)
#     endif()
# else()
#     add_library(simulator_vulkan OBJECT ${SRCS})
#     target_compile_definitions(simulator_vulkan PRIVATE -DVULKAN_HPP_NO_EXCEPTIONS)
#     target_link_libraries(simulator_vulkan PUBLIC nncase)
#     target_link_libraries(simulator_vulkan PRIVATE Vulkan::Vulkan)
#     target_compile_definitions(simulator_vulkan PUBLIC -DNNCASE_MODULES_VULKAN_DLL -DNNCASE_SIMULATOR)
#     set_target_properties(simulator_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
# endif()
