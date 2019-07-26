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
        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const shape_t &perm() const noexcept { return perm_; }

        transpose(datatype_t type, shape_t input_shape, shape_t perm);

        node_opcode opcode() const noexcept override { return op_transpose; }

    private:
        shape_t perm_;
    };
}
}
