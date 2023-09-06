#pragma once

#include "runtime_types.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <gsl/gsl-lite.hpp>
#include <iostream>
#include <linux/limits.h>
#include <numeric>
#include <vector>
#include <nncase/runtime/cpu/compiler_defs.h>


BEGIN_NS_NNCASE_RT_MODULE(cpu)
inline void create_thread(pthread_t &pt, void *param_, void *(*call)(void *))
{
    void *(*new_call)(void *) = call; // + (intptr_t)buf;
    int ret = pthread_create(&pt, NULL, new_call, param_);
    if (ret != 0)
    {
        printf("thread create failed\n");
    }
}

inline void join_thread(pthread_t &pt)
{
    pthread_join(pt, NULL);
}

inline void rt_assert(bool condition, char *message)
{
    assert(condition && message);
}
END_NS_NNCASE_RT_MODULE