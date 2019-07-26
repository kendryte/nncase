#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class dequantize : public node
    {
    public:
        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const quant_param_t quant_param() const noexcept { return quant_param_; }

        dequantize(shape_t input_shape, quant_param_t quant_param);

        node_opcode opcode() const noexcept override { return op_dequantize; }

    private:
        quant_param_t quant_param_;
    };
}
}
