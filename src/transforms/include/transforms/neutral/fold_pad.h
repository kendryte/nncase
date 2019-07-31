#pragma once
#include "..//transform.h"

namespace nncase
{
namespace transforms
{
    class fold_nop_pad_transform : public transform
    {
    public:
        void process(transform_context &context) override;

    protected:
        bool skip_self_contained_check() const noexcept override { return true; }
        bool on_try_match(ir::node &node, transform_context &context) override;
    };

    class fold_pad_strided_slice_transform : public transform
    {
    public:
        void process(transform_context &context) override;

    protected:
        bool skip_self_contained_check() const noexcept override { return true; }
        bool on_try_match(ir::node &node, transform_context &context) override;
    };

    class strided_slice_to_pad_transform : public transform
    {
    public:
        void process(transform_context &context) override;

    protected:
        bool skip_self_contained_check() const noexcept override { return true; }
        bool on_try_match(ir::node &node, transform_context &context) override;
    };
}
}
