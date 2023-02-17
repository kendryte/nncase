
message("****** CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME} ...")
message("****** APPLE: ${APPLE} ...")
message("****** CMAKE_HOST_APPLE: ${CMAKE_HOST_APPLE} ...")
message("****** IOS: ${IOS} ...")

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "(Windows)|(Linux)")
    add_definitions(-DX86_64_SIMD_ON)
endif()

if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
elseif (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    add_compile_options( -mfma -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2)
else()
    message("current platform: other ... ")
endif()
