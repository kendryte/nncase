#include <ir/evaluator.h>
#include <ir/ops/k210/fake_kpu_conv2d.h>
#include <kernels/k210/k210_kernels.h>

using namespace nncase;
using namespace nncase::scheduler;
using namespace nncase::ir;
using namespace nncase::ir::k210;

namespace nncase
{
namespace ir
{
    void register_k210_evaluators()
    {
        register_evaluator(op_k210_fake_kpu_conv2d, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<fake_kpu_conv2d &>(node);

            assert(rnode.input().type() == dt_float32);
            auto input = context.memory_at<float>(rnode.input());
            auto output = context.memory_at<float>(rnode.output());

            auto in_shape = rnode.input().shape();
            shape_t conv_out_shape { in_shape[0], (size_t)rnode.output_channels(), in_shape[2], in_shape[3] };
            auto conv_out_fmap_size = xt::compute_size(conv_out_shape);

            auto conv_output_tmp = std::make_unique<float[]>(conv_out_fmap_size);

#define KPU_CONV2D_IMPL(is_depthwise_val, filter_size_val)                                                                                        \
    if (rnode.is_depthwise() == is_depthwise_val && runtime::k210::get_kpu_filter_size(rnode.filter_type()) == filter_size_val)                                                                       \
    kernels::k210::fake_kpu_conv2d<is_depthwise_val, filter_size_val>(input.data(), conv_output_tmp.get(), rnode.weights().data(), \
        rnode.bias().data(), in_shape[2], in_shape[3], in_shape[1], rnode.output_channels(), rnode.fused_activation())

            KPU_CONV2D_IMPL(true, 1);
            else KPU_CONV2D_IMPL(true, 3);
            else KPU_CONV2D_IMPL(false, 1);
            else KPU_CONV2D_IMPL(false, 3);

            kernels::k210::kpu_pool2d(conv_output_tmp.get(), output.data(), in_shape[2], in_shape[3], rnode.output_channels(),
                rnode.pool_type());
        });
    }
}
}
