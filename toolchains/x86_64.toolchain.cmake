
if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "(Windows)|(Linux)")
    add_definitions(-DX86_64_SIMD_ON)
endif()

if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
    add_compile_options(/arch:AVX)
    add_compile_options(/arch:AVX2)
elseif (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    add_compile_options( -mfma -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2)
else()
    message("current platform: other ... ")
endif()
