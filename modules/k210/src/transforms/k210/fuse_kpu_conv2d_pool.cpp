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
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/k210/kpu_conv2d.h>
#include <nncase/ir/ops/k210/kpu_data_exchange.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/targets/target.h>
#include <nncase/transforms/k210/fuse_kpu_conv2d_pool.h>
#include <nncase/transforms/k210/kpu_utils.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::ir::transforms;
using namespace nncase::ir::transforms::k210;

#define GET_PRE_PAD(reduce)                                                    \
    auto filter_type = get_filter_type(                                        \
        reduce->reduce_op(), reduce->filter_h(), reduce->stride_h());          \
    auto kpu_pad_h = get_kpu_padding(filter_type, reduce->input().shape()[2]); \
    auto kpu_pad_w = get_kpu_padding(filter_type, reduce->input().shape()[3]); \
    padding pad_h{reduce->padding_h().before - kpu_pad_h[0],                   \
                  reduce->padding_h().after - kpu_pad_h[1]};                   \
    padding pad_w{reduce->padding_w().before - kpu_pad_w[0],                   \
                  reduce->padding_w().after - kpu_pad_w[1]};

bool fuse_kpu_conv2d_pool_transform::on_try_match(node &node,
                                                  transform_context &context) {
    if (auto conv = node_cast<kpu_conv2d>(node)) {
        if (!conv->is_depthwise() && conv->pool_type() == kpu_pool_bypass) {
            if (auto kd = try_get_direct_child<kpu_download>(*conv)) {
                if (auto deq = try_get_direct_child<dequantize>(*kd)) {
                    if (auto reduce =
                            try_get_direct_child<reduce_window2d>(*deq)) {
                        if (is_supported_in_shape(reduce->input().shape()) &&
                            is_supported_out_shape(reduce->output().shape()) &&
                            is_supported_filter(
                                reduce->reduce_op(), reduce->filter_h(),
                                reduce->filter_w(), reduce->stride_h(),
                                reduce->stride_w()) &&
                            reduce->dilation_h() == 1 &&
                            reduce->dilation_w() == 1) {
                            GET_PRE_PAD(reduce);

                            if (pad_h == padding::zero() &&
                                pad_w == padding::zero()) {
                                context.inputs.emplace_back(&conv->input());
                                context.inputs.emplace_back(&conv->weights());
                                context.inputs.emplace_back(
                                    &conv->batch_norm());
                                context.inputs.emplace_back(
                                    &conv->activation());
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

void fuse_kpu_conv2d_pool_transform::process(transform_context &context) {
    auto &output = *context.inputs[0]->connection();
    auto &weights = *context.inputs[1]->connection();
    auto &batch_norm = *context.inputs[2]->connection();
    auto &activation = *context.inputs[3]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_conv = static_cast<kpu_conv2d &>(*context.matched_nodes[0]);
    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[2]);
    auto &old_reduce =
        static_cast<reduce_window2d &>(*context.matched_nodes[3]);

    auto pool_type = get_filter_type(
        old_reduce.reduce_op(), old_reduce.filter_h(), old_reduce.stride_h());

    auto conv = context.graph.emplace<kpu_conv2d>(
        old_conv.has_main_mem_output(), old_conv.input().shape(),
        old_conv.is_depthwise(), old_conv.weights().shape(),
        old_conv.filter_type(), pool_type, old_conv.pad_value(),
        old_conv.quant_args(), old_conv.bn(), old_conv.act());
    conv->weights().connect(weights);
    conv->batch_norm().connect(batch_norm);
    conv->activation().connect(activation);
    conv->name(old_conv.name());
    auto kd = context.graph.emplace<kpu_download>(conv->kpu_output().shape());
    kd->name(conv->name() + "/kpu_download");
    auto deq = context.graph.emplace<dequantize>(
        kd->output().type(), kd->output().shape(), dt_float32,
        old_deq.quant_param());
    deq->record_output_connectors_quant_map(deq->output_at(0),
                                            old_reduce.output_at(0));
    deq->record_node_name_before_quant(old_conv.name());
    deq->name(conv->name() + "/dequantize");
    kd->input().connect(conv->kpu_output());
    deq->input().connect(kd->output());

    conv->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
