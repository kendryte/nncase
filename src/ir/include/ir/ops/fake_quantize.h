#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class fake_quantize : public node
    {
    public:
        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        fake_quantize(shape_t input_shape);

        node_opcode opcode() const noexcept override { return op_fake_quantize; }

    private:
    };
}
}
