#pragma once
#include <cassert>

#define NNCASE_CONCAT_3(a, b, c) a / b / c
#define NNCASE_TARGET_HEADER_(target, name) NNCASE_CONCAT_3(runtime, target, name)
#define NNCASE_TARGET_HEADER(name) NNCASE_TARGET_HEADER_(NNCASE_TARGET, name)

#ifndef NNCASE_NO_EXCEPTIONS
#include <stdexcept>
#define NNCASE_THROW(exception, ...) throw exception(__VA_ARGS__)
#else
#define NNCASE_THROW(exception, ...) assert(0 && #exception)
#endif
