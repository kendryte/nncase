include(${CMAKE_CURRENT_LIST_DIR}/riscv64-unknown-elf.toolchain.cmake)

set(K210_SDK_DIR ${K210_SDK_DIR} CACHE STRING "root path to k210 sdk")

set(ENABLE_K210_RUNTIME ON)
set(DEFAULT_BUILTIN_RUNTIMES OFF)
set(DEFAULT_SHARED_RUNTIME_TENSOR_PLATFORM_IMPL OFF)

set(TOOLCHAIN ${RISCV_ROOT_PATH}/bin)
include(${K210_SDK_DIR}/cmake/common.cmake)
include(${K210_SDK_DIR}/cmake/macros.internal.cmake)
include(${CMAKE_SOURCE_DIR}/modules/k210/cmake/kendryte_lib.cmake)