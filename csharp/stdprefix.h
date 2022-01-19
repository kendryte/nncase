#pragma once
#include <limits>
#include <assert.h>
#include <cmath>
#include <cstring>

#define UNUSED(x) (void)(x)
#define DEBUG_ONLY(x) (void)(x)

#ifdef _WIN32
#include <intrin.h>

#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret
#else
#include "unixprefix.h"

#define EXPORT_API(ret) extern "C" __attribute__((visibility("default"))) ret

#define __forceinline __attribute__((always_inline)) inline
#endif