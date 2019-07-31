#pragma once
#include "../transform.h"
#include <ir/quantizer.h>

namespace nncase
{
namespace transforms
{
    namespace k210
    {
        class fuse_kpu_download_transform : public transform
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
