#pragma once
#include "../datatypes.h"
#include "runtime_op.h"

namespace nncase
{
namespace runtime
{
    enum model_target : uint32_t
    {
        MODEL_TARGET_CPU = 0,
        MODEL_TARGET_K210 = 1,
    };

    struct model_header
    {
        uint32_t identifier;
        uint32_t version;
        uint32_t flags;
        model_target target;
        uint32_t constants;
        uint32_t main_mem;
        uint32_t nodes;
        uint32_t inputs;
        uint32_t outputs;
        uint32_t reserved0;
    };

    constexpr uint32_t MODEL_IDENTIFIER = 'KMDL';
    constexpr uint32_t MODEL_VERSION = 4;

    struct node_header
    {
        runtime_opcode opcode;
        uint32_t body_size;
    };
}
}
