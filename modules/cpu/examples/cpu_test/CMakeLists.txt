set(SRC main.cpp)
set(bin cpu_test.elf)

add_executable(${bin} ${SRC})
target_link_libraries(${bin} runtime_cpu)