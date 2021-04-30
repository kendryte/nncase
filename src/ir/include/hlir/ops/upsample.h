#pragma once

#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase::hlir {

    class upsample : public node {

    public:
        DEFINE_NODE_OPCODE(op_upsample)

        upsample(datatype_t dt, shape_t input_shape, const std::vector<float> &scales);

        void compile(hlir_compile_context &context) override;
    private:
        std::vector<float> _scales;
    };
}