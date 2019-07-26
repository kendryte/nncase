#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class reshape : public node
    {
    public:
        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const shape_t &new_shape() const noexcept { return new_shape_; }

        reshape(datatype_t type, shape_t input_shape, shape_t new_shape);

        node_opcode opcode() const noexcept override { return op_reshape; }

    private:
        shape_t new_shape_;
    };
}
}
