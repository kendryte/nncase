#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class unary : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_unary);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        unary_op_t unary_op() const noexcept { return unary_op_; }

        unary(unary_op_t unary_op, shape_t input_shape);

    private:
        unary_op_t unary_op_;
    };
}
}
