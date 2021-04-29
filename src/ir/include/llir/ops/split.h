#pragma once

#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase::llir {

    class split : public node {
    public:
        DEFINE_NODE_OPCODE(op_split)

        split(datatype_t dt, shape_t input_shape, int64_t axis, const std::vector<int64_t>& splits);

        input_connector &input() { return input_at(0); }

        datatype_t type() { return input().type(); }

        const std::vector<int64_t> &splits() { return _splits; }

        int64_t axis() { return _axis; }

    private:
        std::vector<int64_t> _splits;
        int64_t _axis;
    };
}