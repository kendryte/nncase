/* Copyright 2019 Canaan Inc.
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
#include <codegen/codegen.h>
#include <ir/op_utils.h>
#include <ir/ops/k210/kpu_conv2d.h>
#include <ir/ops/k210/kpu_data_exchange.h>
#include <runtime/k210/k210_ops_body.h>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

namespace
{
std::vector<kpu_batchnorm_argument_t> to(const std::vector<kpu_batchnorm_segment> &batch_norm)
{
    std::vector<kpu_batchnorm_argument_t> result(batch_norm.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        auto &src = batch_norm[i];
        auto &dest = result[i];

        dest.batchnorm.data.norm_add = (uint32_t)src.add;
        dest.batchnorm.data.norm_mul = (uint32_t)src.mul;
        dest.batchnorm.data.norm_shift = (uint8_t)src.shift;
    }

    return result;
}

kpu_activate_table_t to(const kpu_activation_table_t &act)
{
    kpu_activate_table_t result;
    for (size_t i = 0; i < 16; i++)
    {
        auto &src = act[i];
        auto &dest = result.activate_para[i];
        auto &bias = i < 8
            ? result.activate_para_bias0.data.result_bias[i]
            : result.activate_para_bias1.data.result_bias[i - 8];

        dest.data.x_start = (uint64_t)src.start_x;
        dest.data.y_mul = (uint16_t)src.mul;
        dest.data.shift_number = (uint8_t)src.shift;
        bias = (uint8_t)src.add;
    }

    return result;
}
}

namespace nncase
{
namespace codegen
{
    void register_k210_emitters()
    {
        register_emitter(op_k210_kpu_upload, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<kpu_upload &>(node);
            auto body = std::make_unique<node_body_impl<rop_k210_kpu_upload, kpu_upload_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());

            return body;
        });

        register_emitter(op_k210_kpu_conv2d, [](node &node, codegen_context &context) {
            struct kpu_conv2d_options_body : public node_body_impl<rop_k210_kpu_conv2d, kpu_conv2d_options>
            {
                std::vector<kpu_batchnorm_argument_t> batch_norm_holder;
                kpu_activate_table_t act_holder;
            };

            auto &rnode = static_cast<kpu_conv2d &>(node);
            auto body = std::make_unique<kpu_conv2d_options_body>();

            auto &layer = body->layer;
            layer = {};
            int32_t one_load_kernel_size, load_times, oc_one_time;
            if (rnode.is_depthwise())
            {
                one_load_kernel_size = rnode.weights().size();
                load_times = 1;
                oc_one_time = rnode.output_channels();
            }
            else
            {
                auto kernel_size = rnode.weights().size();
                auto filter = get_kpu_filter_size(rnode.filter_type());
                auto one_channel_size = filter * filter * rnode.input_channels();
                auto size_limit = 30;
                auto one_load_channels = std::min(rnode.output_channels(), (int32_t)std::floor(size_limit * 1024.f / one_channel_size));
                one_load_kernel_size = one_channel_size * one_load_channels;
                load_times = (int32_t)std::ceil(rnode.output_channels() / (float)one_load_channels);
                oc_one_time = one_load_channels;
            }

            auto in_layout = get_kpu_row_layout(rnode.input().shape()[3]);
            auto out_layout = get_kpu_row_layout(rnode.kpu_output().shape()[3]);

            layer.interrupt_enabe.data.depth_wise_layer = rnode.is_depthwise();
            layer.image_addr.data.image_src_addr = context.get_allocation(rnode.input()).start / 64;
            layer.image_addr.data.image_dst_addr = context.get_allocation(rnode.kpu_output()).start / 64;
            layer.image_channel_num.data.i_ch_num = rnode.input_channels() - 1;
            layer.image_channel_num.data.o_ch_num = rnode.output_channels() - 1;
            layer.image_channel_num.data.o_ch_num_coef = oc_one_time - 1;
            layer.image_size.data.i_col_high = (uint16_t)(rnode.input().shape()[2] - 1);
            layer.image_size.data.i_row_wid = (uint16_t)(rnode.input().shape()[3] - 1);
            layer.image_size.data.o_col_high = (uint16_t)(rnode.kpu_output().shape()[2] - 1);
            layer.image_size.data.o_row_wid = (uint16_t)(rnode.kpu_output().shape()[3] - 1);
            layer.kernel_pool_type_cfg.data.kernel_type = (uint8_t)rnode.filter_type();
            layer.kernel_pool_type_cfg.data.load_para = 1;
            layer.kernel_pool_type_cfg.data.pool_type = (uint8_t)rnode.pool_type();
            layer.kernel_pool_type_cfg.data.dma_burst_size = 15;
            layer.kernel_pool_type_cfg.data.pad_type = 0;
            layer.kernel_pool_type_cfg.data.pad_value = rnode.pad_value();
            layer.kernel_load_cfg.data.load_coor = 1;
            layer.kernel_load_cfg.data.load_time = (uint8_t)(load_times - 1);
            layer.kernel_load_cfg.data.para_size = (uint32_t)one_load_kernel_size;
            layer.kernel_calc_type_cfg.data.channel_switch_addr = (uint16_t)(in_layout.row_len * rnode.input().shape()[2]);
            layer.kernel_calc_type_cfg.data.row_switch_addr = (uint8_t)in_layout.row_len;
            layer.kernel_calc_type_cfg.data.coef_group = (uint8_t)in_layout.groups;
            layer.kernel_calc_type_cfg.data.load_act = 1;
            layer.write_back_cfg.data.wb_channel_switch_addr = (uint16_t)(out_layout.row_len * rnode.kpu_output().shape()[2]);
            layer.write_back_cfg.data.wb_row_switch_addr = (uint8_t)out_layout.row_len;
            layer.write_back_cfg.data.wb_group = (uint8_t)out_layout.groups;
            layer.conv_value.data.arg_x = (uint32_t)rnode.arg_x();
            layer.conv_value.data.shr_x = (uint32_t)rnode.shift_x();
            layer.conv_value.data.arg_w = (uint32_t)rnode.arg_w();
            layer.conv_value.data.shr_w = (uint32_t)rnode.shift_w();
            layer.conv_value2.data.arg_add = (uint64_t)rnode.arg_add();
            layer.dma_parameter.data.channel_byte_num = (uint16_t)(rnode.kpu_output().shape()[3] * rnode.kpu_output().shape()[2] - 1);
            layer.dma_parameter.data.dma_total_byte = (uint32_t)(rnode.kpu_output().shape()[3] * rnode.kpu_output().shape()[2] * rnode.kpu_output().shape()[1] - 1);
            layer.dma_parameter.data.send_data_out = rnode.has_main_mem_output();

            body->main_mem_output = rnode.has_main_mem_output() ? context.get_allocation(rnode.main_mem_output()) : memory_range {};
            body->batches = (int32_t)rnode.input().shape()[0];
            body->batch_norm_holder = to(rnode.batch_norm());
            body->act_holder = to(rnode.activation());
            body->batch_norm = body->batch_norm_holder;
            body->activation = &body->act_holder;
            body->weights = rnode.weights();

            return body;
        });
    }
}
}
