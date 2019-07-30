#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class concat : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_concat);

        output_connector &output() { return output_at(0); }

        int32_t axis() const noexcept { return axis_; }
        xtl::span<const int32_t> concat_dims() const noexcept { return concat_dims_; }

        concat(datatype_t type, xtl::span<shape_t> input_shapes, int32_t axis);

    private:
        int32_t axis_;
        std::vector<int32_t> concat_dims_;
    };
}
}
