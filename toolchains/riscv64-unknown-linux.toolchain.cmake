set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(DEFINED ENV{RISCV_ROOT_PATH})
    file(TO_CMAKE_PATH $ENV{RISCV_ROOT_PATH} RISCV_ROOT_PATH)
endif()

if(NOT RISCV_ROOT_PATH)
    message(FATAL_ERROR "RISCV_ROOT_PATH env must be defined")
endif()

set(RISCV_ROOT_PATH ${RISCV_ROOT_PATH} CACHE STRING "root path to riscv toolchain")

set(CMAKE_C_COMPILER "${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++")

set(CMAKE_FIND_ROOT_PATH "${RISCV_ROOT_PATH}/riscv64-unknown-linux-gnu")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(ENABLE_VULKAN_RUNTIME OFF)
set(ENABLE_OPENMP OFF)
set(ENABLE_VULKAN OFF)
set(ENABLE_HALIDE OFF)
set(BUILD_PYTHON_BINDING OFF)
# -march=rv64gv_zvl128b_zvfh -mrvv-vector-bits=zvl
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv_zfh_zvfh -mabi=lp64d -mcmodel=medany -fstack-protector-strong -fPIE -pie -Wl,-z,now -Wl,-z,relro -mrvv-v0p10-compatible")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv_zfh_zvfh -mabi=lp64d -mcmodel=medany -fstack-protector-strong -fPIE -pie -Wl,-z,now -Wl,-z,relro  -mrvv-v0p10-compatible")