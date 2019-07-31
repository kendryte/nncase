#pragma once
#include "..//transform.h"

namespace nncase
{
namespace transforms
{
    namespace k210
    {
        class fake_kpu_conv2d_transform : public transform
        {
        public:
            void process(transform_context &context) override;

        protected:
            bool skip_self_contained_check() const noexcept override { return true; }
            bool on_try_match(ir::node &node, transform_context &context) override;
        };

        class fuse_fake_kpu_conv2d_strided_slice_transform : public transform
        {
        public:
            void process(transform_context &context) override;

        protected:
            bool skip_self_contained_check() const noexcept override { return true; }
            bool on_try_match(ir::node &node, transform_context &context) override;
        };
    }
}
}
