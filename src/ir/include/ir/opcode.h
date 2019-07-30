#pragma once
#include <stdexcept>
#include <string_view>

namespace nncase
{
namespace ir
{
#define DEFINE_NEUTRAL_OPCODE(id, name, value) op_##id = value,
#define DEFINE_OPCODE(target, id, name, value) op_##target_##id = value,

    enum node_opcode
    {
#include "opcode.def"
    };

#undef DEFINE_NEUTRAL_OPCODE
#undef DEFINE_OPCODE
#define DEFINE_NEUTRAL_OPCODE(id, name, value) \
    case op_##id:                              \
        return #name;
#define DEFINE_OPCODE(target, id, name, value) \
    case op_##target_##id:                     \
        return #name;

    constexpr std::string_view node_opcode_names(node_opcode opcode)
    {
        switch (opcode)
        {
#include "opcode.def"
        default:
            throw std::invalid_argument("invalid opcode");
        }
    }

#undef DEFINE_NEUTRAL_OPCODE
#undef DEFINE_OPCODE
}
}
