#pragma once
#include "../node.h"
#include <runtime/runtime_op_utility.h>
#include <vector>

namespace nncase
{
namespace ir
{
    class constant : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_constant);

        output_connector &output() { return output_at(0); }

        xtl::span<const uint8_t> data() const noexcept { return data_; }

        template <class TShape, class... TDataArgs>
        constant(datatype_t type, TShape &&shape, TDataArgs... data_args)
            : data_(std::forward<TDataArgs>(data_args)...)
        {
            add_output("output", type, std::forward<TShape>(shape), mem_const);
        }

        template <class TScalar>
        constant(TScalar scalar)
            : data_(reinterpret_cast<const uint8_t *>(&scalar), reinterpret_cast<const uint8_t *>(&scalar) + sizeof(scalar))
        {
            add_output("output", runtime::to_datatype_v<TScalar>, shape_t { 1 }, mem_const);
        }

    private:
        std::vector<uint8_t> data_;
    };
}
}
