#pragma once
#include "..//transform.h"

namespace nncase
{
namespace transforms
{
    class fold_transpose_transform : public transform
    {
    public:
        void process(transform_context &context) override;
    protected:
        bool skip_self_contained_check() const noexcept override { return true; }
        bool on_try_match(ir::node &node, transform_context &context) override;
    };

    class fold_nop_transpose_transform : public transform
    {
    public:
        void process(transform_context &context) override;

    protected:
        bool skip_self_contained_check() const noexcept override { return true; }
        bool on_try_match(ir::node &node, transform_context &context) override;
    };
}
}
