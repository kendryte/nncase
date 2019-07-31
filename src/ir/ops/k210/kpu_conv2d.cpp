#include <ir/op_utils.h>
#include <ir/ops/k210/kpu_conv2d.h>
#include <runtime/k210/k210_runtime_op_utility.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;

kpu_conv2d::kpu_conv2d(bool has_main_mem_output, shape_t input_shape, bool is_depthwise, runtime::k210::kpu_filter_type_t filter_type, runtime::k210::kpu_pool_type_t pool_type, xt::xtensor<uint8_t, 4> weights,
    uint8_t pad_value, int32_t arg_x, int32_t shift_x, int32_t arg_w, int32_t shift_w, int64_t arg_add, std::vector<runtime::k210::kpu_batchnorm_segment> batch_norm, runtime::k210::kpu_activation_table_t activation)
    : weights_(std::move(weights)), is_depthwise_(is_depthwise), filter_type_(filter_type), pool_type_(pool_type), pad_value_(pad_value), arg_x_(arg_x), shift_x_(shift_x), arg_w_(arg_w), shift_w_(shift_w), arg_add_(arg_add), batch_norm_(std::move(batch_norm)), activation_(activation)
{
    add_input("input", dt_uint8, input_shape);
    add_output("output", dt_uint8,
        shape_t {
            input_shape[0],
            (size_t)output_channels(),
            (size_t)get_kpu_pool_output_size((int32_t)input_shape[2], pool_type_),
            (size_t)get_kpu_pool_output_size((int32_t)input_shape[3], pool_type_) },
        mem_k210_kpu);

    if (has_main_mem_output)
        add_output("main_mem_output", dt_uint8, kpu_output().shape());
}
