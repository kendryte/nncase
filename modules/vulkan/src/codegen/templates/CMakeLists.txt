# cmake_minimum_required (VERSION 3.16)

# function(compress_template NAME FILES OUT_VAR)
#     if(NOT LIBZIP_ZIPTOOL_EXECUTABLE)
#         set(LIBZIP_ZIPTOOL_EXECUTABLE ziptool)
#     endif()
#     set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.xz)
#     set(SCRIPT_FILE ${CMAKE_SOURCE_DIR}/tools/compress_files.py)
#     if(WIN32)
#         set(PYTHON_EXECUTABLE "python")
#     else()
#         set(PYTHON_EXECUTABLE "python3")
#     endif()
#     add_custom_command(
#         OUTPUT ${OUT_FILE}
#         COMMAND ${PYTHON_EXECUTABLE}
#         ARGS ${SCRIPT_FILE} -t ${LIBZIP_ZIPTOOL_EXECUTABLE} -o ${OUT_FILE} ${FILES}
#         DEPENDS ${FILES}
#         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#         VERBATIM)
#     add_custom_target(${NAME}
#         DEPENDS ${OUT_FILE})
#     set(${OUT_VAR} ${OUT_FILE} PARENT_SCOPE)
# endfunction()

# set(TEMPLATE_SRCS unary.hlsl
#                   unary.comp)
# compress_template(vulkan_templates_xz "${TEMPLATE_SRCS}" TEMPLATE_OUT)

# if(MSVC)
#     add_library(vulkan_templates_rc OBJECT Resource.rc)
#     target_include_directories(vulkan_templates_rc PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
#     set_source_files_properties(
#         Resource.rc
#         PROPERTIES
#         OBJECT_DEPENDS ${TEMPLATE_OUT}
#     )
# endif()

# set(SRCS template.cpp)
# add_library(vulkan_templates STATIC ${SRCS})
# target_link_libraries(vulkan_templates PRIVATE libzippp::libzippp pantor::inja nlohmann_json::nlohmann_json fmt::fmt shaderc::shaderc)
# set_target_properties(vulkan_templates PROPERTIES POSITION_INDEPENDENT_CODE ON)
# add_dependencies(vulkan_templates vulkan_templates_xz)
# target_include_directories(vulkan_templates PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# if(MSVC)
#     target_compile_options(vulkan_templates PRIVATE /wd4244) #inja
# else()
#     set_source_files_properties(
#         template.cpp
#         PROPERTIES
#         OBJECT_DEPENDS ${TEMPLATE_OUT}
#     )
# endif()