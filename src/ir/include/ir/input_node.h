#pragma once 
#include "node.h"

namespace nncase
{
namespace ir
{
    class input_node : public node
    {
    public:
        output_connector &output() { return output_at(0); }

        template <class TShape>
        input_node(datatype_t type, TShape &&shape)
        {
            add_output("output", type, std::forward<TShape>(shape));
        }

        node_opcode opcode() const noexcept override { return op_input; }
    };
}
}
