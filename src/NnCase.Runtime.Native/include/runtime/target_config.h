#pragma once

#define NNCASE_CONCAT_3(a, b, c) a/b/c
#define NNCASE_TARGET_HEADER_(target, name) <NNCASE_CONCAT_3(targets, target, name)>
#define NNCASE_TARGET_HEADER(name) NNCASE_TARGET_HEADER_(NNCASE_TARGET, name)

#include NNCASE_TARGET_HEADER(interpreter.h)

namespace nncase
{
namespace runtime
{
    using interpreter_t = nncase::targets::NNCASE_TARGET::interpreter;
}
}
