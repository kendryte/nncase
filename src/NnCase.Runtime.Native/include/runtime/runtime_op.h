#pragma once
#include "..//datatypes.h"

namespace nncase
{
namespace runtime
{
#define DEFINE_RUNTIME_OP(id, name, value) rop_##id = value,

    enum runtime_opcode : uint32_t
    {
#include "runtime_op.def"
    };

#undef DEFINE_RUNTIME_OP
}
}
