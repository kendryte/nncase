#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class softmax : public node
    {
    public:
        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        float beta() const noexcept { return beta_; }

        softmax(shape_t input_shape, float beta);

        node_opcode opcode() const noexcept override { return op_softmax; }

    private:
        float beta_;
    };
}
}
