cmake_minimum_required (VERSION 3.13)

if (WIN32)
    set(SRCS pe/pe_loader.cpp)
elseif (APPLE)
    set(SRCS macho/macho_loader.cpp)
else ()
    set(SRCS elf/elf_loader.cpp
             elf/elfload.cpp
             elf/elfreloc_aarch64.cpp
             elf/elfreloc_amd64.cpp
             elf/elfreloc_i386.cpp
             elf/elfreloc_riscv64.cpp)
endif()

add_library(cpu_loaders OBJECT ${SRCS})
set_property(TARGET cpu_loaders PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cpu_loaders PUBLIC gsl::gsl-lite)

if (BUILDING_RUNTIME)
    install(TARGETS cpu_loaders EXPORT nncaseruntimeTargets)
endif()
