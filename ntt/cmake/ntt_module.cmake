cmake_minimum_required(VERSION 3.16)

include(${CMAKE_CURRENT_LIST_DIR}/compile_flags.cmake)

if (CMAKE_CUDA_COMPILER)
    set(NNCASE_NTT_MODULE_TARGET_NAME nncase_ntt_module_bundle)

    find_program(NVLINK nvlink REQUIRED)
    find_program(FATBINARY fatbinary REQUIRED)
    message(STATUS "Found nvlink: ${NVLINK}")
    message(STATUS "Found fatbinary: ${FATBINARY}")
else()
    set(NNCASE_NTT_MODULE_TARGET_NAME nncase_ntt_module)
endif()

if (BUILD_STANDALONE)
    add_executable(${NNCASE_NTT_MODULE_TARGET_NAME} ${CMAKE_CURRENT_LIST_DIR}/../src/dummy.cpp)
elseif (CMAKE_CUDA_COMPILER)
    add_library(${NNCASE_NTT_MODULE_TARGET_NAME} OBJECT)
else()
    add_library(${NNCASE_NTT_MODULE_TARGET_NAME} SHARED ${CMAKE_CURRENT_LIST_DIR}/../src/dummy.cpp)
endif()

target_compile_features(${NNCASE_NTT_MODULE_TARGET_NAME} PUBLIC cxx_std_20)
target_include_directories(${NNCASE_NTT_MODULE_TARGET_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)
set_target_properties(${NNCASE_NTT_MODULE_TARGET_NAME} PROPERTIES PREFIX "" SUFFIX "")
set_target_properties(${NNCASE_NTT_MODULE_TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${NNCASE_NTT_MODULE_TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

if (BUILD_STANDALONE)
    target_compile_definitions(${NNCASE_NTT_MODULE_TARGET_NAME} PUBLIC -DNNCASE_STANDALONE=1)
endif()

if (MSVC)
    set_property(TARGET ${NNCASE_NTT_MODULE_TARGET_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set_target_properties(${NNCASE_NTT_MODULE_TARGET_NAME} PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE)
    target_link_options(${NNCASE_NTT_MODULE_TARGET_NAME} PRIVATE /NODEFAULTLIB)
    target_link_libraries(${NNCASE_NTT_MODULE_TARGET_NAME} PRIVATE "libvcruntime$<$<CONFIG:Debug>:d>"
                                                    "msvcrt$<$<CONFIG:Debug>:d>"
                                                    "ucrt$<$<CONFIG:Debug>:d>"
                                                    "libcpmt$<$<CONFIG:Debug>:d>")
elseif(APPLE)
    target_link_options(${NNCASE_NTT_MODULE_TARGET_NAME} PRIVATE -ld_classic -lc)
else()
    target_link_libraries(${NNCASE_NTT_MODULE_TARGET_NAME} PRIVATE pthread)
endif()

if (CMAKE_CUDA_COMPILER)
    target_sources(${NNCASE_NTT_MODULE_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src/cuda_runtime.cu)
    target_compile_definitions(${NNCASE_NTT_MODULE_TARGET_NAME} PUBLIC -DNNCASE_CUDA_MODULE=1)
    target_compile_options(${NNCASE_NTT_MODULE_TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -fgpu-rdc
        --cuda-device-only
        >)

    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        # Link device code for this architecture
        set(linked_obj "${CMAKE_CURRENT_BINARY_DIR}/linked_sm_${arch}.o")
        add_custom_command(
            OUTPUT ${linked_obj}
            COMMAND ${NVLINK}
                -arch=sm_${arch}
                $<TARGET_OBJECTS:${NNCASE_NTT_MODULE_TARGET_NAME}>
                -o ${linked_obj}
            DEPENDS ${NNCASE_NTT_MODULE_TARGET_NAME}
            COMMAND_EXPAND_LISTS
            VERBATIM
            COMMENT "Linking device code for sm_${arch}"
        )

        # Add to the list of all linked objects
        list(APPEND ALL_LINKED_OBJECTS ${linked_obj})
    endforeach()

    add_custom_target(device_link ALL
        DEPENDS ${ALL_LINKED_OBJECTS}
    )

    set(FATBIN_ARGS "")
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        # Find the linked object for this architecture
        set(arch_obj "${CMAKE_CURRENT_BINARY_DIR}/linked_sm_${arch}.o")
        list(APPEND FATBIN_ARGS --image3=kind=elf,sm=${arch},file="${arch_obj}")
    endforeach()

    # Create the fatbinary from all linked objects
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nncase_ntt_module
        COMMAND ${FATBINARY}
            -64
            --create ${CMAKE_CURRENT_BINARY_DIR}/nncase_ntt_module
            ${FATBIN_ARGS}
        DEPENDS ${ALL_LINKED_OBJECTS}
        COMMENT "Creating fatbinary from linked objects"
        VERBATIM
    )

    add_custom_target(fatbin ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/nncase_ntt_module
    )
else()
    target_sources(${NNCASE_NTT_MODULE_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src/cpu_runtime.cpp)
    target_compile_definitions(${NNCASE_NTT_MODULE_TARGET_NAME} PUBLIC -DNNCASE_CPU_MODULE=1)
endif()
