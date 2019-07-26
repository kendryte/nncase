#pragma once
#include "target_interpreter.h"
#include <datatypes.h>
#include <runtime/runtime_op.h>
#include <xtl/xspan.hpp>

namespace nncase
{
namespace runtime
{
    enum kernel_call_result
    {
        kcr_done,
        kcr_async,
        kcr_error
    };

    kernel_call_result call_kernel(runtime_opcode opcode, xtl::span<const uint8_t> body, interpreter_t &interpreter, interpreter_step_t step);
}
}
