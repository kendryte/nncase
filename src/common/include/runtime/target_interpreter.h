#pragma once
#include "../target_config.h"

// clang-format off
#include NNCASE_TARGET_HEADER(runtime,interpreter.h)
// clang-format on

namespace nncase
{
namespace runtime
{
    using interpreter_t = nncase::runtime::NNCASE_TARGET::interpreter;
}
}
