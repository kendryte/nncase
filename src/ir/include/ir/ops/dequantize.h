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
        DEFINE_NODE_OPCODE(op_dequantize);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const quant_param_t quant_param() const noexcept { return quant_param_; }

        dequantize(shape_t input_shape, quant_param_t quant_param);

    private:
        quant_param_t quant_param_;
    };
}
}
