#pragma once
#include "../datatypes.h"
#include "binary_writer.h"
#include "span_reader.h"

namespace nncase
{
namespace runtime
{
    template <class T>
    struct simple_node_body
    {
        void deserialize(span_reader &reader)
        {
            reader.read(static_cast<T &>(*this));
        }

        void serialize(binary_writer &writer) const
        {
            writer.write(static_cast<const T &>(*this));
        }
    };
}
}
