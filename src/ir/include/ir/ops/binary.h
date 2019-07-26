#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class binary : public node
    {
    public:
        input_connector &input_a() { return input_at(0); }
        input_connector &input_b() { return input_at(1); }
        output_connector &output() { return output_at(0); }

        binary_op_t binary_op() const noexcept { return binary_op_; }
        value_range<float> fused_activation() const noexcept { return fused_activation_; }

        binary(binary_op_t binary_op, shape_t input_a_shape, shape_t input_b_shape, value_range<float> fused_activation);

        node_opcode opcode() const noexcept override { return op_binary; }

    private:
        binary_op_t binary_op_;
        value_range<float> fused_activation_;
    };
}
}
