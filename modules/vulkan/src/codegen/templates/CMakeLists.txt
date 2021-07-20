cmake_minimum_required (VERSION 3.13)

function(compress_template NAME FILES)
    if(NOT LIBZIP_ZIPTOOL_EXECUTABLE)
        set(LIBZIP_ZIPTOOL_EXECUTABLE ziptool)
    endif()
    set(ADD_FILES )
    set(INDEX 0)
    foreach(FILE ${FILES})
        get_filename_component(FULL_PATH "${FILE}"
                               REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        get_filename_component(FILE_NAME "${FILE}" NAME)
        file (SIZE ${FULL_PATH} FILE_SIZE)
        list(APPEND ADD_FILES add_file "${FILE_NAME}" "${FULL_PATH}" 0 ${FILE_SIZE})
        list(APPEND ADD_FILES set_file_compression ${INDEX} deflate 0)
        MATH(EXPR INDEX "${INDEX}+1")
    endforeach()
    set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.xz)
    add_custom_command(
        OUTPUT ${OUT_FILE}
        COMMAND ${LIBZIP_ZIPTOOL_EXECUTABLE}
        ARGS -nt ${OUT_FILE} ${ADD_FILES}
        DEPENDS ${FILES}
        VERBATIM)
    add_custom_target(${NAME}
        SOURCES ${OUT_FILE})
endfunction()

set(TEMPLATE_SRCS 1.comp)
compress_template(vulkan_templates_xz "${TEMPLATE_SRCS}")

if(MSVC)
    add_library(vulkan_templates_rc INTERFACE)
    target_sources(vulkan_templates_rc PUBLIC Resource.rc)
    target_include_directories(vulkan_templates_rc INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
    add_dependencies(vulkan_templates_rc vulkan_templates_xz)
endif()

set(SRCS template.cpp)
add_library(vulkan_templates OBJECT ${SRCS})
target_link_libraries(vulkan_templates PRIVATE libzippp::libzippp)
set_target_properties(vulkan_templates PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(vulkan_templates vulkan_templates_xz)
target_include_directories(vulkan_templates PRIVATE ${CMAKE_CURRENT_BINARY_DIR})