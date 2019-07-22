#include <runtime/kernel_registry.h>
#include <runtime/span_reader.h>
#include <targets/cpu/cpu_ops_body.h>
#include <targets/k210/k210_ops_body.h>
#include <targets/neutral/neutral_ops_body.h>

using namespace nncase;
using namespace nncase::runtime;

namespace nncase
{
namespace targets
{
#define BEGINE_DEFINE_TARGET(target) \
    namespace target                 \
    {

#define DEFINE_RUNTIME_OP(target, id, name, value) \
    kernel_call_result id(id##_options &, interpreter_t &, interpreter_step_t);

#define END_DEFINE_TARGET() }

#include <runtime/runtime_op.def>

#undef BEGINE_DEFINE_TARGET
#undef DEFINE_RUNTIME_OP
#undef END_DEFINE_TARGET
}
}

kernel_call_result runtime::call_kernel(runtime_opcode opcode, xtl::span<const uint8_t> body, interpreter_t &interpreter, interpreter_step_t step)
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
