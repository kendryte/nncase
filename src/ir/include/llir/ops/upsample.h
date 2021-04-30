#pragma once

#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase::llir {

    class upsample : public node {
    public:
        DEFINE_NODE_OPCODE(op_upsample)

        upsample(datatype_t dt, shape_t input_shape, const std::vector<float> &scales);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }
        datatype_t type() { return input().type(); }
        const std::vector<float> &scales() { return _scales; }
    private:
        std::vector<float> _scales;
    };
}