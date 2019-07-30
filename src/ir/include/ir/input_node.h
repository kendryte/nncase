#pragma once 
#include "node.h"

namespace nncase
{
namespace ir
{
    class input_node : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_input_node);

        output_connector &output() { return output_at(0); }

        template <class TShape>
        input_node(datatype_t type, TShape &&shape)
        {
            add_output("output", type, std::forward<TShape>(shape));
        }
    };
}
}
