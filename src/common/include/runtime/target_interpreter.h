#pragma once
#include "../target_config.h"

#include NNCASE_TARGET_HEADER(interpreter.h)

namespace nncase
{
namespace runtime
{
    using interpreter_t = nncase::runtime::NNCASE_TARGET::interpreter;
}
}
