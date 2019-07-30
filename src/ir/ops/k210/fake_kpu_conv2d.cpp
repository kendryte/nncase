#include <ir/op_utils.h>
#include <ir/ops/k210/fake_kpu_conv2d.h>
#include <runtime/k210/k210_runtime_op_utility.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;

fake_kpu_conv2d::fake_kpu_conv2d(shape_t input_shape, bool is_depthwise, runtime::k210::kpu_filter_type_t filter_type, runtime::k210::kpu_pool_type_t pool_type, xt::xtensor<float, 4> weights, xt::xtensor<float, 1> bias, value_range<float> fused_activation)
    : weights_(std::move(weights)), bias_(std::move(bias)), is_depthwise_(is_depthwise), filter_type_(filter_type), pool_type_(pool_type), fused_activation_(fused_activation)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32,
        shape_t {
            input_shape[0],
            (size_t)output_channels(),
            (size_t)get_kpu_pool_output_size((int32_t)input_shape[2], pool_type_),
            (size_t)get_kpu_pool_output_size((int32_t)input_shape[3], pool_type_) });
}
