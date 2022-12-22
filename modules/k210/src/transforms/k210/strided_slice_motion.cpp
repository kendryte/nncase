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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/fused_unary.h>
#include <nncase/ir/ops/k210/fake_kpu_conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/transforms/k210/strided_slice_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::ir::transforms;
using namespace nncase::ir::transforms::k210;

bool strided_slice_motion_transform::on_try_match(node &node,
                                                  transform_context &context) {
    if (node.runtime_opcode() == op_slice) {
        auto &slc = static_cast<slice &>(node);
        if (slc.strides() == axis_t{1, 1, 2, 2} &&
            slc.begin() == axis_t{0, 0, 0, 0} &&
            slc.end() == axis_t{(int32_t)slc.input().shape()[0],
                                (int32_t)slc.input().shape()[1],
                                (int32_t)slc.input().shape()[2],
                                (int32_t)slc.input().shape()[3]} &&
            slc.begin_mask() == 0 && slc.end_mask() == 0 &&
            slc.new_axis_mask() == 0) {
            if (auto conv = try_get_direct_child<fake_kpu_conv2d>(slc)) {
                if (!conv->is_depthwise() &&
                    conv->filter_type() == kpu_filter_1x1 &&
                    conv->pool_type() == kpu_pool_bypass) {
                    context.inputs.emplace_back(&slc.input());
                    context.inputs.emplace_back(&conv->weights());
                    context.inputs.emplace_back(&conv->bias());
                    context.outputs.emplace_back(&conv->output());

                    context.matched_nodes.emplace_back(&slc);
                    context.matched_nodes.emplace_back(conv);
                    return true;
                }
            }
        }
    }

    return false;
}

void strided_slice_motion_transform::process(transform_context &context) {
    auto &output = *context.inputs[0]->connection();
    auto &weights = *context.inputs[1]->connection();
    auto &bias = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_slice = static_cast<slice &>(*context.matched_nodes[0]);
    auto &old_conv = static_cast<fake_kpu_conv2d &>(*context.matched_nodes[1]);

    auto conv = context.graph.emplace<fake_kpu_conv2d>(
        output.shape(), old_conv.is_depthwise(), old_conv.weights().shape(),
        old_conv.filter_type(), old_conv.pool_type(),
        old_conv.fused_activation());
    conv->name(old_conv.name());
    conv->weights().connect(weights);
    conv->bias().connect(bias);
    auto slc = context.graph.emplace<slice>(
        dt_float32, conv->output().shape(), axis_t{0, 0, 0, 0},
        axis_t{0, 0, 0, 0}, old_slice.strides(), 15, 15, 0, 0);
    slc->input().connect(conv->output());

    conv->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(slc->output());
}

/**
 * @brief add conv2d 1x1 before (top_left, top_right, bottom_left, bottom_right)
 * 2x2 slice, for kpu compute.
 *
 */
bool strided_slice_conv2d_pool::on_try_match(node &node,
                                             transform_context &context) {
    if (node.runtime_opcode() == op_slice) {
        // || slc.begin() == axis_t { 0, 0, 1, 0 } || slc.begin() == axis_t { 0,
        // 0, 1, 1 }
        auto &slc = static_cast<slice &>(node);
        if ((slc.strides() == axis_t{1, 1, 2, 2}) &&
            (slc.begin() == axis_t{0, 0, 0, 0} ||
             slc.begin() == axis_t{0, 0, 0, 1}) &&
            slc.end() == axis_t{(int32_t)slc.input().shape()[0],
                                (int32_t)slc.input().shape()[1],
                                (int32_t)slc.input().shape()[2],
                                (int32_t)slc.input().shape()[3]} &&
            slc.begin_mask() == 0 && slc.end_mask() == 0 &&
            slc.new_axis_mask() == 0) {
            if (try_get_direct_parent<input_node>(slc)) {
                context.inputs.emplace_back(&slc.input());
                context.outputs.emplace_back(&slc.output());

                context.matched_nodes.emplace_back(&slc);
                return true;
            }
        }
    }

    return false;
}

void strided_slice_conv2d_pool::process(transform_context &context) {
    auto &last_output = *context.inputs[0]->connection();
    auto next_input = context.outputs[0]->connections();

    auto &old_slice = static_cast<slice &>(*context.matched_nodes[0]);

    shape_t weights_shape{3, 3, 1, 1};
    std::vector<float> weights_value{1.f, 0.f, 0.f, 0.f, 1.f,
                                     0.f, 0.f, 0.f, 1.f};
    auto weights = context.graph.emplace<constant>(dt_float32, weights_shape,
                                                   weights_value);
    weights->name(old_slice.name() + "/Pre_Conv/Weights");

    shape_t bias_shape{3};
    std::vector<float> bias_value(3, 0.f);
    auto bias =
        context.graph.emplace<constant>(dt_float32, bias_shape, bias_value);
    bias->name(old_slice.name() + "/Pre_Conv/Bias");

    int32_t stride_h, stride_w;
    padding padding_h, padding_w;
    axis_t begins, ends{0, 0, 0, 0}, strides;

    if (old_slice.begin() == axis_t{0, 0, 0, 0} ||
        old_slice.begin() == axis_t{0, 0, 0, 1}) {
        /* top left and top right */
        stride_h = stride_w = 1;
        padding_h = padding_w = padding::zero();

        begins = old_slice.begin();
        strides = old_slice.strides();
    } else if (old_slice.begin() == axis_t{0, 0, 1, 0}) {
        /* bot_left */
        stride_h = stride_w = 2;
        padding_h = padding{1, 1};
        padding_w = padding::zero();
        begins = axis_t{0, 0, 1, 0};
        strides = axis_t{1, 1, 1, 1};
    } else if (old_slice.begin() == axis_t{0, 0, 1, 1}) {
        /* bot_right */
        stride_h = stride_w = 2;
        padding_h = padding{1, 1};
        padding_w = padding{1, 1};
        begins = axis_t{0, 0, 1, 1};
        strides = axis_t{1, 1, 1, 1};
    } else {
        throw std::runtime_error("the slice begin not support ");
    }

    auto conv = context.graph.emplace<conv2d>(
        last_output.shape(), weights_shape, 1, padding_h, padding_w, stride_h,
        stride_w, 1, 1, value_range<float>::full());
    conv->name(old_slice.name() + "/Pre_Conv");
    conv->weights().connect(weights->output());
    conv->bias().connect(bias->output());

    for (size_t i = 0; i < conv->output().shape().size(); i++) {
        ends[i] = (int32_t)conv->output().shape()[i];
    }

    auto slc = context.graph.emplace<slice>(
        dt_float32, conv->output().shape(), begins, ends, strides,
        old_slice.begin_mask(), old_slice.end_mask(), old_slice.ellipsis_mask(),
        old_slice.new_axis_mask());
    slc->name(old_slice.name());
    slc->input().connect(conv->output());

    conv->input().connect(last_output);
    for (auto &in : dup(next_input))
        in->connect(slc->output());
}
