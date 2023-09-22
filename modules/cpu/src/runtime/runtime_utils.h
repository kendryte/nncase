#pragma once

#include "gsl-lite.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#if defined(__linux__)
#include <linux/limits.h>
#elif defined(__APPLE__)
#include <pthread.h>
#endif
#include <nncase/runtime/cpu/compiler_defs.h>
#include <numeric>
#include <vector>

BEGIN_NS_NNCASE_RT_MODULE(cpu)
inline void create_thread(pthread_t &pt, void *param_, void *(*call)(void *)) {
    void *(*new_call)(void *) = call; // + (intptr_t)buf;
    int ret = pthread_create(&pt, NULL, new_call, param_);
    if (ret != 0) {
        printf("thread create failed\n");
    }
}

inline void join_thread(pthread_t &pt) { pthread_join(pt, NULL); }

inline void rt_assert([[maybe_unused]] bool condition,
                      [[maybe_unused]] char *message) {
    assert(condition && message);
}
END_NS_NNCASE_RT_MODULE