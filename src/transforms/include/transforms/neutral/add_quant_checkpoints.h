#pragma once
#include "..//transform.h"
#include <unordered_set>

namespace nncase
{
namespace transforms
{
    class add_quant_checkpoints_transform : public transform
    {
    public:
        add_quant_checkpoints_transform(std::initializer_list<ir::node_opcode>&& opcodes)
            : opcodes_(std::move(opcodes))
        {
        }

        void process(transform_context &context) override;

    protected:
        bool skip_self_contained_check() const noexcept override { return true; }
        bool on_try_match(ir::node &node, transform_context &context) override;

    private:
        std::unordered_set<ir::node_opcode> opcodes_;
    };
}
}
