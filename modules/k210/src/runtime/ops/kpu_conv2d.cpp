/* Copyright 2019-2020 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../runtime_module.h"
#include <nncase/kernels/k210/k210_kernels.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

result<void> k210_runtime_module::visit(const kpu_conv2d_options &op) noexcept
{
    auto &layer = op.layer;
    auto in_h = static_cast<uint32_t>(layer.image_size.data.i_col_high + 1);
    auto in_w = static_cast<uint32_t>(layer.image_size.data.i_row_wid + 1);
    auto in_ch = static_cast<uint32_t>(layer.image_channel_num.data.i_ch_num + 1);

    auto out_h = static_cast<uint32_t>(layer.image_size.data.o_col_high + 1);
    auto out_w = static_cast<uint32_t>(layer.image_size.data.o_row_wid + 1);
    auto out_ch = static_cast<uint32_t>(layer.image_channel_num.data.o_ch_num + 1);

    auto is_depthwise = layer.interrupt_enabe.data.depth_wise_layer != 0;

    try_var(input, memory_at({ .memory_location = mem_kpu, .datatype = dt_uint8, .start = (uint32_t)layer.image_addr.data.image_src_addr * 64, .size = 1 }));
    try_var(weights, memory_at(op.weights));
    try_var(batch_norm_data, memory_at(op.batch_norm));
    try_var(activation_data, memory_at(op.activation));
    try_var(kpu_out, memory_at({ .memory_location = mem_kpu, .datatype = dt_uint8, .start = (uint32_t)layer.image_addr.data.image_dst_addr * 64, .size = 1 }));

#ifdef NNCASE_SIMULATOR
    kpu_shape_t in_shape { op.batches, in_ch, in_h, in_w };
    kpu_shape_t out_shape { op.batches, out_ch, out_h, out_w };
    auto in_fmap_size = kernels::detail::compute_size(in_shape);

    kpu_shape_t conv_out_shape { op.batches, out_ch, in_h, in_w };
    auto conv_out_fmap_size = kernels::detail::compute_size(conv_out_shape);
    auto out_fmap_size = kernels::detail::compute_size(out_shape);

    auto input_tmp = std::make_unique<uint8_t[]>(in_fmap_size);
    auto workspace = std::make_unique<int64_t[]>(conv_out_fmap_size);
    auto conv_output_tmp = std::make_unique<uint8_t[]>(conv_out_fmap_size);
    auto output_tmp = std::make_unique<uint8_t[]>(out_fmap_size);

    auto batch = in_shape[0];
    auto in_size_per_batch = kernels::detail::compute_size(in_shape) / batch;
    auto conv_output_tmp_size_per_batch = conv_out_fmap_size / batch;
    auto out_size_per_batch = kernels::detail::compute_size(out_shape) / batch;
    auto p_input = input_tmp.get();
    auto p_workspace = workspace.get();
    auto p_conv_ouput_tmp = conv_output_tmp.get();
    auto p_output_tmp = output_tmp.get();

    try_(kernels::k210::kpu_download(reinterpret_cast<const uint8_t *>(input.data()), input_tmp.get(), in_shape));
    auto filter_size = get_kpu_filter_size((kpu_filter_type_t)layer.kernel_pool_type_cfg.data.kernel_type);
    auto pad_value = (uint8_t)layer.kernel_pool_type_cfg.data.pad_value;
    auto arg_x = (int32_t)kernels::detail::to_signed<24>(layer.conv_value.data.arg_x);
    auto shift_x = (int32_t)layer.conv_value.data.shr_x;
    auto arg_w = (int32_t)kernels::detail::to_signed<24>(layer.conv_value.data.arg_w);
    auto shift_w = (int32_t)layer.conv_value.data.shr_w;
    auto arg_add = kernels::detail::to_signed<40>(layer.conv_value2.data.arg_add);

    auto batchnorm = std::make_unique<kpu_batchnorm_segment[]>(out_ch);
    for (size_t i = 0; i < out_ch; i++)
    {
        auto &src = batch_norm_data.as_span<const kpu_batchnorm_argument_t>()[i].batchnorm.data;
        auto &dest = batchnorm[i];
        dest.mul = (int32_t)kernels::detail::to_signed<24>(src.norm_mul);
        dest.shift = (int32_t)src.norm_shift;
        dest.add = (int32_t)kernels::detail::to_signed<32>(src.norm_add);
    }

    kpu_activation_table_t activation;
    for (size_t i = 0; i < 16; i++)
    {
        auto &act_table = activation_data.as_span<const kpu_activate_table_t>()[0];
        auto &src = act_table.activate_para[i].data;
        auto &dest = activation[i];
        dest.start_x = kernels::detail::to_signed<36>(src.x_start);
        dest.mul = (int32_t)kernels::detail::to_signed<16>(src.y_mul);
        dest.shift = (int32_t)src.shift_number;

        if (i < 16)
            dest.add = act_table.activate_para_bias0.data.result_bias[i];
        else
            dest.add = act_table.activate_para_bias1.data.result_bias[i - 16];
    }

#define KPU_CONV2D_IMPL(is_depthwise_val, filter_size_val)                                                                                                  \
    if (is_depthwise == is_depthwise_val && filter_size == filter_size_val)                                                                                 \
    kernels::k210::kpu_conv2d<is_depthwise_val, filter_size_val>(p_input, p_workspace, p_conv_ouput_tmp, reinterpret_cast<const uint8_t *>(weights.data()), \
        in_h, in_w, in_ch, out_ch, pad_value, arg_x, shift_x, arg_w, shift_w, arg_add, batchnorm.get(), activation)

    for (size_t n = 0; n < batch; n++)
    {
        KPU_CONV2D_IMPL(true, 1);
        else KPU_CONV2D_IMPL(true, 3);
        else KPU_CONV2D_IMPL(false, 1);
        else KPU_CONV2D_IMPL(false, 3);

        kernels::k210::kpu_pool2d(p_conv_ouput_tmp, p_output_tmp, in_h, in_w, out_ch, (kpu_pool_type_t)layer.kernel_pool_type_cfg.data.pool_type);

        p_input += in_size_per_batch;
        p_workspace += conv_output_tmp_size_per_batch;
        p_conv_ouput_tmp += conv_output_tmp_size_per_batch;
        p_output_tmp += out_size_per_batch;
    }

    try_(kernels::k210::kpu_upload(output_tmp.get(), reinterpret_cast<uint8_t *>(kpu_out.data()), out_shape));
    if (op.main_mem_output.size)
    {
        try_var(main_output, memory_at(op.main_mem_output));
        std::copy(output_tmp.get(), output_tmp.get() + out_fmap_size, reinterpret_cast<uint8_t *>(main_output.data()));
    }
#endif
    return ok();
}
