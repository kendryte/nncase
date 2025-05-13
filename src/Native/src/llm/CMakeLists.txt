cmake_minimum_required (VERSION 3.8)

set(SRCS attention_kv_cache.cpp
         paged_attention_kv_cache.cpp
         paged_attention_scheduler.cpp
)

if (BUILDING_RUNTIME)
    add_library(llm OBJECT ${SRCS})
    target_include_directories(llm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    set_property(TARGET llm PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS llm EXPORT nncaseruntimeTargets)
else()
    add_library(llm OBJECT ${SRCS})
    target_include_directories(llm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    
    target_compile_definitions(llm PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
    set_property(TARGET llm PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(APPLE)
    target_compile_options(llm PRIVATE -Wno-gnu-zero-variadic-macro-arguments)
endif(APPLE)

