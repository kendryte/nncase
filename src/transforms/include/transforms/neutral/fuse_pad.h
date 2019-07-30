#pragma once
#include "..//transform.h"

namespace nncase
{
namespace transforms
{
#define DEFINE_FP_MOTION(name)                                                    \
    class fuse_pad_##name##_transform : public transform                          \
    {                                                                             \
    public:                                                                       \
        void process(transform_context &context) override;                        \
                                                                                  \
    protected:                                                                    \
        bool skip_self_contained_check() const noexcept override { return true; } \
        bool on_try_match(ir::node &node, transform_context &context) override;   \
    };

    DEFINE_FP_MOTION(conv2d);

#undef DEFINE_FP_MOTION
}
}
