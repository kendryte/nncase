#pragma once
#include "../datatypes.h"
#include <string_view>

namespace nncase
{
namespace runtime
{
#define BEGINE_DEFINE_TARGET(...)
#define DEFINE_NEUTRAL_RUNTIME_OP(id, name, value) rop_##id = value,
#define DEFINE_RUNTIME_OP(target, id, name, value) rop_##target##_##id = value,
#define END_DEFINE_TARGET()

    enum runtime_opcode : uint32_t
    {
#include "runtime_op.def"
    };

#undef DEFINE_NEUTRAL_RUNTIME_OP
#undef DEFINE_RUNTIME_OP
#define DEFINE_NEUTRAL_RUNTIME_OP(id, name, value) \
    case rop_##id:                                 \
        return #name;
#define DEFINE_RUNTIME_OP(target, id, name, value) \
    case rop_##target##_##id:                      \
        return #name;

    constexpr std::string_view node_opcode_names(runtime_opcode opcode)
    {
        switch (opcode)
        {
#include "runtime_op.def"
        default:
            return {};
        }
    }

#undef BEGINE_DEFINE_TARGET
#undef DEFINE_NEUTRAL_RUNTIME_OP
#undef DEFINE_RUNTIME_OP
#undef END_DEFINE_TARGET
}
}
