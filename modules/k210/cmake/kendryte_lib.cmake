file(GLOB_RECURSE LIB_SRC
        "${K210_SDK_DIR}/lib/bsp/*.h"
        "${K210_SDK_DIR}/lib/drivers/*.h"
        "${K210_SDK_DIR}/lib/utils/*.h"
        "${K210_SDK_DIR}/lib/bsp/*.hpp"
        "${K210_SDK_DIR}/lib/drivers/*.hpp"
        "${K210_SDK_DIR}/lib/utils/*.hpp"
        "${K210_SDK_DIR}/lib/bsp/*.c"
        "${K210_SDK_DIR}/lib/drivers/*.c"
        "${K210_SDK_DIR}/lib/utils/*.c"
        "${K210_SDK_DIR}/lib/bsp/*.cpp"
        "${K210_SDK_DIR}/lib/drivers/*.cpp"
        "${K210_SDK_DIR}/lib/utils/*.cpp"
        "${K210_SDK_DIR}/lib/bsp/*.s"
        "${K210_SDK_DIR}/lib/drivers/*.s"
        "${K210_SDK_DIR}/lib/utils/*.s"
        "${K210_SDK_DIR}/lib/bsp/*.S"
        "${K210_SDK_DIR}/lib/drivers/*.S"
        "${K210_SDK_DIR}/lib/utils/*.S"
        )

FILE(GLOB_RECURSE ASSEMBLY_FILES
        "${K210_SDK_DIR}/lib/bsp/*.s"
        "${K210_SDK_DIR}/lib/drivers/*.s"
        "${K210_SDK_DIR}/lib/utils/*.s"
        "${K210_SDK_DIR}/lib/bsp/*.S"
        "${K210_SDK_DIR}/lib/drivers/*.S"
        "${K210_SDK_DIR}/lib/utils/*.S"
        )

set_property(SOURCE ${ASSEMBLY_FILES} PROPERTY LANGUAGE C)
set_source_files_properties(${ASSEMBLY_FILES} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp -D __riscv64")

add_library(kendryte STATIC ${LIB_SRC})
target_include_directories(kendryte PUBLIC
    ${K210_SDK_DIR}/lib/drivers/include
    ${K210_SDK_DIR}/lib/bsp/include
    ${K210_SDK_DIR}/lib/utils/include
    ${K210_SDK_DIR}/lib/nncase/include)
target_compile_options(kendryte PRIVATE -Wno-multichar)
install(TARGETS kendryte EXPORT nncaseruntimeTargets)

function(target_link_kendryte TARGET)
    target_link_libraries(${TARGET} PRIVATE kendryte)
    get_target_property(TARGET_TYPE ${TARGET} TYPE)
    if (TARGET_TYPE STREQUAL "EXECUTABLE")
        get_target_property(TARGET_OUT_NAME ${TARGET} OUTPUT_NAME)
        set(BIN_NAME $<TARGET_FILE_PREFIX:${TARGET}>$<TARGET_FILE_PREFIX:${TARGET}>$<TARGET_FILE_BASE_NAME:${TARGET}>.bin)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_OBJCOPY} --output-format=binary $<TARGET_FILE:${TARGET}> $<TARGET_FILE_DIR:${TARGET}>/${BIN_NAME}
            DEPENDS ${TARGET}
            COMMENT "Generating .bin file for ${TARGET}...")
    endif()
endfunction()
