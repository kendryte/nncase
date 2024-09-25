# cmake_minimum_required (VERSION 3.8)

# add_subdirectory(templates)

# set(SRCS module_builder.cpp
#          ops/unary.cpp
#          ops/copy.cpp)

# add_library(codegen_vulkan OBJECT ${SRCS})
# target_link_libraries(codegen_vulkan PUBLIC nncase)
# target_link_libraries(codegen_vulkan PRIVATE vulkan_templates Vulkan::Headers)
# target_compile_definitions(codegen_vulkan PUBLIC -DNNCASE_MODULES_VULKAN_DLL)
# set_target_properties(codegen_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
