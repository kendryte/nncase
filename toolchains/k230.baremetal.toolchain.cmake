set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(DEFINED ENV{RISCV_ROOT_PATH})
    file(TO_CMAKE_PATH $ENV{RISCV_ROOT_PATH} RISCV_ROOT_PATH)
endif()

if(NOT RISCV_ROOT_PATH)
    message(FATAL_ERROR "RISCV_ROOT_PATH env must be defined")
endif()

set(RISCV_ROOT_PATH ${RISCV_ROOT_PATH} CACHE STRING "root path to riscv toolchain")

set(CMAKE_C_COMPILER "${RISCV_ROOT_PATH}/bin/riscv64-unknown-elf-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_ROOT_PATH}/bin/riscv64-unknown-elf-g++")

set(CMAKE_FIND_ROOT_PATH "${RISCV_ROOT_PATH}/riscv64-unknown-elf")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -static")   #-march=rv64imafdc_v0p7_zfh_zvamo0p7_zvlsseg0p7_xtheadc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64imafdcv -mabi=lp64d -mcmodel=medany -static") 

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(ENABLE_VULKAN_RUNTIME OFF)
set(ENABLE_OPENMP OFF)
set(ENABLE_VULKAN OFF)
set(ENABLE_HALIDE OFF)
set(DEFAULT_BUILTIN_RUNTIMES OFF)
set(BUILD_PYTHON_BINDING OFF)
set(DEFAULT_SHARED_RUNTIME_TENSOR_PLATFORM_IMPL OFF)
set(BUILD_BENCHMARK OFF)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64imafdcv_zihintpause_zfh_zba_zbb_zbc_zbs_xtheadc -mabi=lp64d -mcmodel=medany -mtune=c908")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64imafdcv_zihintpause_zfh_zba_zbb_zbc_zbs_xtheadc -mabi=lp64d -mcmodel=medany -mtune=c908")