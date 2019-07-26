#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class matmul : public node
    {
    public:
        input_connector &input_a() { return input_at(0); }
        input_connector &input_b() { return input_at(1); }
        output_connector &output() { return output_at(0); }

        const xt::xtensor<float, 1> &bias() const noexcept { return bias_; }
        value_range<float> fused_activation() const noexcept { return fused_activation_; }

        matmul(shape_t input_a_shape, shape_t input_b_shape, xt::xtensor<float, 1> bias, value_range<float> fused_activation);

        node_opcode opcode() const noexcept override { return op_matmul; }

    private:
        xt::xtensor<float, 1> bias_;
        value_range<float> fused_activation_;
    };
}
}
