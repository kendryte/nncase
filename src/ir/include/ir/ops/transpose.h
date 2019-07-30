#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class transpose : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_transpose);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const axis_t &perm() const noexcept { return perm_; }

        transpose(datatype_t type, shape_t input_shape, axis_t perm);

    private:
        axis_t perm_;
    };
}
}
