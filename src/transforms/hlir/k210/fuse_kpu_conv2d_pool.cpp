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
#include <hlir/ops/dequantize.h>
#include <hlir/ops/k210/kpu_conv2d.h>
#include <hlir/ops/k210/kpu_data_exchange.h>
#include <hlir/ops/quantize.h>
#include <hlir/ops/reduce_window2d.h>
#include <hlir/transforms/k210/fuse_kpu_conv2d_pool.h>
#include <hlir/transforms/k210/kpu_utils.h>
#include <hlir/visitor.h>
#include <runtime/k210/k210_runtime_op_utility.h>
#include <targets/target.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::hlir::transforms;
using namespace nncase::hlir::transforms::k210;

#define GET_PRE_PAD(reduce)                                                                                      \
    auto filter_type = hlir::k210::get_filter_type(reduce->reduce_op(), reduce->filter_h(), reduce->stride_h()); \
    auto kpu_pad_h = get_kpu_padding(filter_type, reduce->input().shape()[2]);                                   \
    auto kpu_pad_w = get_kpu_padding(filter_type, reduce->input().shape()[3]);                                   \
    padding pad_h { reduce->padding_h().before - kpu_pad_h[0], reduce->padding_h().after - kpu_pad_h[1] };       \
    padding pad_w { reduce->padding_w().before - kpu_pad_w[0], reduce->padding_w().after - kpu_pad_w[1] };       \
                                                                                                                 \
    auto pre_pad_h = hlir::k210::get_padding<true>(pad_h);                                                       \
    auto pre_pad_w = hlir::k210::get_padding<true>(pad_w);

bool fuse_kpu_conv2d_pool_transform::on_try_match(node &node, transform_context &context)
{
    if (auto conv = node_cast<kpu_conv2d>(node))
    {
        if (!conv->is_depthwise() && conv->pool_type() == kpu_pool_bypass)
        {
            if (auto kd = try_get_direct_child<kpu_download>(*conv))
            {
                if (auto deq = try_get_direct_child<dequantize>(*kd))
                {
                    if (auto reduce = try_get_direct_child<reduce_window2d>(*deq))
                    {
                        if (hlir::k210::is_supported_in_shape(reduce->input().shape())
                            && hlir::k210::is_supported_out_shape(reduce->output().shape())
                            && hlir::k210::is_supported_filter(reduce->reduce_op(), reduce->filter_h(), reduce->filter_w(), reduce->stride_h(), reduce->stride_w())
                            && reduce->dilation_h() == 1 && reduce->dilation_w() == 1)
                        {
                            GET_PRE_PAD(reduce);

                            if (pad_h == padding::zero() && pad_w == padding::zero())
                            {
                                context.inputs.emplace_back(&conv->input());
                                context.outputs.emplace_back(&reduce->output());

                                context.matched_nodes.emplace_back(conv);
                                context.matched_nodes.emplace_back(kd);
                                context.matched_nodes.emplace_back(deq);
                                context.matched_nodes.emplace_back(reduce);
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    return false;
}

void fuse_kpu_conv2d_pool_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_conv = static_cast<kpu_conv2d &>(*context.matched_nodes[0]);
    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[2]);
    auto &old_reduce = static_cast<reduce_window2d &>(*context.matched_nodes[3]);

    auto pool_type = hlir::k210::get_filter_type(old_reduce.reduce_op(), old_reduce.filter_h(), old_reduce.stride_h());

    auto conv = context.graph.emplace<kpu_conv2d>(old_conv.has_main_mem_output(), old_conv.input().shape(), old_conv.is_depthwise(), old_conv.filter_type(), pool_type,
        old_conv.weights(), old_conv.pad_value(), old_conv.arg_x(), old_conv.shift_x(), old_conv.arg_w(), old_conv.shift_w(), old_conv.arg_add(), old_conv.batch_norm(), old_conv.activation());
    conv->name(old_conv.name());
    auto kd = context.graph.emplace<kpu_download>(conv->kpu_output().shape());
    kd->name(conv->name() + "/kpu_download");
    auto deq = context.graph.emplace<dequantize>(kd->output().shape(), old_deq.quant_param());
    deq->name(conv->name() + "/dequantize");
    kd->input().connect(conv->kpu_output());
    deq->input().connect(kd->output());

    conv->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
