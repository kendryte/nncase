#pragma once
#include <string_view>
#include <stdexcept>

#define DEFINE_OPCODE(name, value) op_##name = value,

namespace nncase
{
namespace ir
{
    enum node_opcode
    {
#include "opcode.def"
    };
}
}

#undef DEFINE_OPCODE
#define DEFINE_OPCODE(name, value) \
    case op_##name:                \
        return #name;

namespace nncase
{
namespace ir
{
    constexpr std::string_view node_opcode_names(node_opcode opcode)
    {
        switch (opcode)
        {
#include "opcode.def"
        default:
            throw std::invalid_argument("invalid opcode");
        }
    }
}
}

#undef DEFINE_OPCODE
