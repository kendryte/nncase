if(${CMAKE_SYSTEM_PROCESSOR} MATCHES
   "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
   include(${CMAKE_CURRENT_LIST_DIR}/x86_64.toolchain.cmake)
endif()
