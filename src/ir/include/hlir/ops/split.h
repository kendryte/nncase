#pragma once

#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase::hlir {

    class split : public node {

    public:
        DEFINE_NODE_OPCODE(op_split)

        split(datatype_t dt, shape_t input_shape, int64_t axis, const std::vector<int64_t>& splits);

        void compile(hlir_compile_context &context) override;

        input_connector &input() { return input_at(0); }
    private:
        int64_t _axis;
        std::vector<int64_t> _splits;
    };
}