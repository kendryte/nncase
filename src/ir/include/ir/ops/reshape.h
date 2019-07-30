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
        DEFINE_NODE_OPCODE(op_reshape);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const shape_t &new_shape() const noexcept { return new_shape_; }

        reshape(datatype_t type, shape_t input_shape, axis_t new_shape);
        reshape(datatype_t type, shape_t input_shape, shape_t new_shape);

    private:
        shape_t new_shape_;
    };
}
}
