#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class pad : public node
    {
    public:
        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const xt::svector<padding> &paddings() const noexcept { return paddings_; }
        const scalar &pad_value() const noexcept { return pad_value_; }

        pad(datatype_t type, shape_t input_shape, xt::svector<padding> paddings, scalar pad_value);

        node_opcode opcode() const noexcept override { return op_pad; }

    private:
        xt::svector<padding> paddings_;
        scalar pad_value_;
    };
}
}
