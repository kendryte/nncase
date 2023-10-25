set(SRC main.cpp)
set(bin xpu_test.elf)

add_executable(${bin} ${SRC})
target_link_libraries(${bin} runtime_xpu)