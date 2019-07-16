#pragma once
#include "../../runtime/kernel_registry.h"
#include "../../runtime/span_reader.h"
#include <runtime/op_utility.h>

namespace nncase
{
namespace targets
{
    template <runtime::runtime_opcode Op, class T>
    struct simple_node_body
    {
        void deserialize(runtime::span_reader &reader)
        {
            reader.read(static_cast<T &>(*this));
        }
    };
}
}
