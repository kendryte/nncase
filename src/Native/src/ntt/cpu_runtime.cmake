cmake_minimum_required(VERSION 3.15)

if (MSVC)
    add_definitions(/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DUNICODE /D_UNICODE)
    add_compile_options(/Zc:threadSafeInit- /utf-8 /wd4200 /Oi)
    # Disable C++ exceptions.
    string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c- /GS-")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /GS-")
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    add_definitions(-D_HAS_EXCEPTIONS=0)

    # Disable RTTI.
    string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
    add_compile_options(/arch:AVX /fp:fast)
else()
    add_compile_options(-ffast-math -flto -Wno-multichar -Wno-unused-value -fno-common -ffunction-sections -fno-exceptions -fdata-sections -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-stack-protector)
    add_link_options(-flto)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

    if (APPLE)
        add_compile_options(-fno-stack-check)
    endif()
endif()

add_library(nncase_cpu_runtime STATIC ${CMAKE_CURRENT_LIST_DIR}/cpu_runtime.cpp)
target_compile_features(nncase_cpu_runtime PUBLIC cxx_std_20)
target_include_directories(nncase_cpu_runtime PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)

if (MSVC)
    set_property(TARGET nncase_cpu_runtime PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
