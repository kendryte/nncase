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

    class output_node : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_output_node);

        input_connector &input() { return input_at(0); }

        template <class TShape>
        output_node(datatype_t type, TShape &&shape)
        {
            add_input("input", type, std::forward<TShape>(shape));
        }
    };

    class ignore_node : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_ignore_node);

        input_connector &input() { return input_at(0); }

        template <class TShape>
        ignore_node(datatype_t type, TShape &&shape)
        {
            add_input("input", type, std::forward<TShape>(shape));
        }

        node_attributes attributes() const noexcept override { return node_attr_action; }
    };
}
}
