cmake_minimum_required (VERSION 3.13)

if (BUILDING_RUNTIME)
    if (ENABLE_K210_RUNTIME)
        if (NOT K210_SDK_DIR)
            message(FATAL_ERROR "Please set K210_SDK_DIR to kendryte sdk root path")
        endif()
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
        target_compile_options(kendryte PRIVATE -Wno-multichar -Wno-pointer-arith -Wno-unused-parameter -Wno-pedantic
                                                -Wno-sign-compare -Wno-implicit-fallthrough -Wno-strict-aliasing)
        install(TARGETS kendryte EXPORT nncaseruntimeTargets)

    endif()
endif()
