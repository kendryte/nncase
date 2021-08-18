// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_ALLOCATOR_H
#define NCNN_ALLOCATOR_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <stdlib.h>

namespace ncnn
{

#if __AVX__
// the alignment of all the allocated buffers
#define MALLOC_ALIGN 32
#else
// the alignment of all the allocated buffers
#define MALLOC_ALIGN 16
#endif

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template <typename _Tp>
static inline _Tp *alignPtr(_Tp *ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp *)(((size_t)ptr + n - 1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static inline void *fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void *ptr = 0;
    if (posix_memalign(&ptr, MALLOC_ALIGN, size))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(MALLOC_ALIGN, size);
#else
    unsigned char *udata = (unsigned char *)malloc(size + sizeof(void *) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char **adata = alignPtr((unsigned char **)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}

static inline void fastFree(void *ptr)
{
    if (ptr)
    {
#if _MSC_VER
        _aligned_free(ptr);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
        free(ptr);
#elif __ANDROID__ && __ANDROID_API__ < 17
        free(ptr);
#else
        unsigned char *udata = ((unsigned char **)ptr)[-1];
        free(udata);
#endif
    }
}

#if NCNN_THREADS
// exchange-add operation for atomic operations on reference counters
#if defined __riscv && !defined __riscv_atomic
// riscv target without A extension
static inline int NCNN_XADD(int *addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
#elif defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void *>(reinterpret_cast<volatile void *>(addr)), delta)
#elif defined __GNUC__
#if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#ifdef __ATOMIC_ACQ_REL
#define NCNN_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int) *)(addr), delta, __ATOMIC_ACQ_REL)
#else
#define NCNN_XADD(addr, delta) __atomic_fetch_add((_Atomic(int) *)(addr), delta, 4)
#endif
#else
#if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#define NCNN_XADD(addr, delta) (int)__atomic_fetch_add((unsigned *)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#else
#define NCNN_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned *)(addr), (unsigned)(delta))
#endif
#endif
#elif defined _MSC_VER && !defined RC_INVOKED
#define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile *)addr, delta)
#else
// thread-unsafe branch
static inline int NCNN_XADD(int *addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
#endif
#else // NCNN_THREADS
static inline int NCNN_XADD(int *addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
#endif // NCNN_THREADS

class Allocator
{
public:
    virtual ~Allocator();
    virtual void *fastMalloc(size_t size) = 0;
    virtual void fastFree(void *ptr) = 0;
};

} // namespace ncnn

#endif // NCNN_ALLOCATOR_H
