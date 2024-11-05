cmake_minimum_required(VERSION 3.15)

# Disable C++ exceptions & RTTI
if (MSVC)
    add_definitions(/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DUNICODE /D_UNICODE)
    add_compile_options(/Zc:threadSafeInit- /utf-8 /wd4200 /Oi)
    string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c- /GS-")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /GS-")
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    add_definitions(-D_HAS_EXCEPTIONS=0)

    string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
else()
    add_compile_options(-Wno-multichar -Wno-unused-value -fno-common -ffunction-sections -fno-exceptions -fdata-sections -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-stack-protector)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

    if (APPLE)
        add_compile_options(-fno-stack-check -Wno-c++11-narrowing)
    else()
        add_compile_options(-Wnarrowing)
    endif()
endif()

# fp contract
if (MSVC)
    add_compile_options(/fp:contract)
else()
    add_compile_options(-ffp-contract=on)
endif()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES
   "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
    if (MSVC)
        add_compile_options(/arch:AVX2)
        add_compile_definitions(__SSE2__ __SSE4_1__ __FMA__ __AVX__ __AVX2__)
    else()
        add_compile_options(-mavx2 -mfma)
    endif()
endif()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES
   "arm64|ARM64|aarch64.*|AARCH64.*")
endif()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES
   "riscv64")
   if(ENABLE_K230_RUNTIME)
       add_compile_options(-march=rv64gv_zvl128b_zfh -mrvv-vector-bits=zvl)
   elseif(ENABLE_K80_RUNTIME)
       add_compile_options(-march=rv64gcv_zvl1024b_zfh -mrvv-vector-bits=zvl)
   else()
       message(FATAL_ERROR "Unsupported riscv64 target")
   endif()
endif()
