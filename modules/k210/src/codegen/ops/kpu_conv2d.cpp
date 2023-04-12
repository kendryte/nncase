/* Copyright 2019-2021 Canaan Inc.
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
#include "../module_builder.h"
#include <nncase/ir/ops/k210/runtime_type_utils.h>
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/runtime/k210/runtime_types.h>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::k210;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

void k210_module_builder::emit(kpu_conv2d &node) {
    kpu_conv2d_options options{};
    auto &layer = options.layer;
    layer = {};
    int32_t one_load_kernel_size, load_times, oc_one_time;
    if (node.is_depthwise()) {
        one_load_kernel_size = xt::compute_size(node.weights().shape());
        load_times = 1;
        oc_one_time = node.output_channels();
    } else {
        auto filter = get_kpu_filter_size(node.filter_type());
        auto one_channel_size = filter * filter * node.input_channels();
        auto size_limit = 30;
        auto one_load_channels = std::min(
            node.output_channels(),
            (int32_t)std::floor(size_limit * 1024.f / one_channel_size));
        one_load_kernel_size = one_channel_size * one_load_channels;
        load_times = (int32_t)std::ceil(node.output_channels() /
                                        (float)one_load_channels);
        oc_one_time = one_load_channels;
    }

    auto in_layout = get_kpu_row_layout(node.input().shape()[3]);
    auto out_layout = get_kpu_row_layout(node.kpu_output().shape()[3]);

    layer.interrupt_enabe.data.depth_wise_layer = node.is_depthwise();
    layer.image_addr.data.image_src_addr = allocation(node.input()).start / 64;
    layer.image_addr.data.image_dst_addr =
        allocation(node.kpu_output()).start / 64;
    layer.image_channel_num.data.i_ch_num = node.input_channels() - 1;
    layer.image_channel_num.data.o_ch_num = node.output_channels() - 1;
    layer.image_channel_num.data.o_ch_num_coef = oc_one_time - 1;
    layer.image_size.data.i_col_high = (uint16_t)(node.input().shape()[2] - 1);
    layer.image_size.data.i_row_wid = (uint16_t)(node.input().shape()[3] - 1);
    layer.image_size.data.o_col_high =
        (uint16_t)(node.kpu_output().shape()[2] - 1);
    layer.image_size.data.o_row_wid =
        (uint16_t)(node.kpu_output().shape()[3] - 1);
    layer.kernel_pool_type_cfg.data.kernel_type = (uint8_t)node.filter_type();
    layer.kernel_pool_type_cfg.data.load_para = 1;
    layer.kernel_pool_type_cfg.data.pool_type = (uint8_t)node.pool_type();
    layer.kernel_pool_type_cfg.data.dma_burst_size = 15;
    layer.kernel_pool_type_cfg.data.pad_type = 0;
    layer.kernel_pool_type_cfg.data.pad_value = node.pad_value();
    layer.kernel_load_cfg.data.load_coor = 1;
    layer.kernel_load_cfg.data.load_time = (uint8_t)(load_times - 1);
    layer.kernel_load_cfg.data.para_size = (uint32_t)one_load_kernel_size;
    layer.kernel_calc_type_cfg.data.channel_switch_addr =
        (uint16_t)(in_layout.row_len * node.input().shape()[2]);
    layer.kernel_calc_type_cfg.data.row_switch_addr =
        (uint8_t)in_layout.row_len;
    layer.kernel_calc_type_cfg.data.coef_group = (uint8_t)in_layout.groups;
    layer.kernel_calc_type_cfg.data.load_act = 1;
    layer.write_back_cfg.data.wb_channel_switch_addr =
        (uint16_t)(out_layout.row_len * node.kpu_output().shape()[2]);
    layer.write_back_cfg.data.wb_row_switch_addr = (uint8_t)out_layout.row_len;
    layer.write_back_cfg.data.wb_group = (uint8_t)out_layout.groups;
    layer.conv_value.data.arg_x = (uint32_t)node.quant_args().arg_x;
    layer.conv_value.data.shr_x = (uint32_t)node.quant_args().shift_x;
    layer.conv_value.data.arg_w = (uint32_t)node.quant_args().arg_w;
    layer.conv_value.data.shr_w = (uint32_t)node.quant_args().shift_w;
    layer.conv_value2.data.arg_add = (uint64_t)node.quant_args().arg_add;
    layer.dma_parameter.data.channel_byte_num = (uint16_t)(
        node.kpu_output().shape()[3] * node.kpu_output().shape()[2] - 1);
    layer.dma_parameter.data.dma_total_byte =
        (uint32_t)(node.kpu_output().shape()[3] * node.kpu_output().shape()[2] *
                       node.kpu_output().shape()[1] -
                   1);
    layer.dma_parameter.data.send_data_out = node.has_main_mem_output();

    options.weights = allocation(node.weights()).runtime_type();
    options.batch_norm = allocation(node.batch_norm()).runtime_type();
    options.activation = allocation(node.activation()).runtime_type();
    options.main_mem_output =
        node.has_main_mem_output()
            ? allocation(node.main_mem_output()).runtime_type()
            : memory_range{};
    options.batches = (int32_t)node.input().shape()[0];
    text_writer().write(options);
}
