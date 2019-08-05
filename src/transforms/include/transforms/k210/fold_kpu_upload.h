#pragma once
#include "../transform.h"
#include <ir/quantizer.h>

namespace nncase
{
namespace transforms
{
    namespace k210
    {
        class fold_kpu_upload_transform : public transform
        {
        public:
            void process(transform_context &context) override;

        protected:
            bool skip_self_contained_check() const noexcept override { return true; }
            bool on_try_match(ir::node &node, transform_context &context) override;
        };

        class fold_input_kpu_upload_transform : public transform
        {
        public:
            void process(transform_context &context) override;

        protected:
            bool on_try_match(ir::node &node, transform_context &context) override;
        };
    }
}
}
