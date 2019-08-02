#pragma once
#include "..//transform.h"

namespace nncase
{
namespace transforms
{
    class fold_quantize_transform : public transform
    {
    public:
        void process(transform_context &context) override;

    protected:
        bool skip_self_contained_check() const noexcept override { return true; }
        bool on_try_match(ir::node &node, transform_context &context) override;
    };

    class fold_input_quantize_transform : public transform
    {
    public:
        fold_input_quantize_transform(quant_param_t quant_param)
            : quant_param_(quant_param)
        {
        }

        void process(transform_context &context) override;

    protected:
        bool on_try_match(ir::node &node, transform_context &context) override;

    private:
        quant_param_t quant_param_;
    };
}
}
