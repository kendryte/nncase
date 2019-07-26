#pragma once 
#include "../node.h"
#include <vector>

namespace nncase
{
namespace ir
{
    class constant : public node
    {
    public:
        output_connector &output() { return output_at(0); }

        xtl::span<const uint8_t> data() const noexcept { return data_; }

        template <class TShape>
        constant(datatype_t type, TShape &&shape, std::vector<uint8_t> data)
            : data_(std::move(data))
        {
            add_output("output", type, std::forward<TShape>(shape));
        }

        node_opcode opcode() const noexcept override { return op_constant; }

    private:
        std::vector<uint8_t> data_;
    };
}
}
