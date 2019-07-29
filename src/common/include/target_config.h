#pragma once
#include <cassert>

// clang-format off
#define NNCASE_CONCAT_3(a, b, c) a/b/c
// clang-format on

#define NNCASE_TARGET_HEADER_(prefix, target, name) <NNCASE_CONCAT_3(prefix, target, name)>
#define NNCASE_TARGET_HEADER(prefix, name) NNCASE_TARGET_HEADER_(prefix, NNCASE_TARGET, name)

#ifndef NNCASE_NO_EXCEPTIONS
#include <stdexcept>
#define NNCASE_THROW(exception, ...) throw exception(__VA_ARGS__)
#else
#define NNCASE_THROW(exception, ...) assert(0 && #exception)
#endif
