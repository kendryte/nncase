#include "kernel_registry.h"
#include "../targets/cpu/cpu_ops.h"
#include "../targets/neutral/neutral_ops.h"
#include "span_reader.h"

using namespace nncase;
using namespace nncase::runtime;

kernel_call_result runtime::call_kernel(runtime_opcode opcode, xtl::span<const uint8_t> body, interpreter &interpreter, interpreter_step_t step)
{
    span_reader reader(body);

    switch (opcode)
    {
#define BEGINE_DEFINE_TARGET(...)
#define DEFINE_RUNTIME_OP(target, id, name, value)                      \
    case rop_##id:                                                      \
    {                                                                   \
        nncase::targets::target::id##_options options;                  \
        options.deserialize(reader);                                    \
        return nncase::targets::target::id(options, interpreter, step); \
    }
#define END_DEFINE_TARGET()

#include <runtime/runtime_op.def>

#undef BEGINE_DEFINE_TARGET
#undef DEFINE_RUNTIME_OP
#undef END_DEFINE_TARGET
    default:
        return kcr_error;
    }
}
