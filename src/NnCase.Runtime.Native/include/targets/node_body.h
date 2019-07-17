#pragma once
#include "../runtime/binary_writer.h"
#include "../runtime/span_reader.h"
#include <datatypes.h>

namespace nncase
{
namespace targets
{
    template <class T>
    struct simple_node_body
    {
        void deserialize(runtime::span_reader &reader)
        {
            reader.read(static_cast<T &>(*this));
        }

        void serialize(runtime::binary_writer &writer) const
        {
            writer.write(static_cast<const T &>(*this));
        }
    };
}
}
