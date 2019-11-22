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
#include <ir/ops/binary.h>
#include <ir/ops/concat.h>
#include <ir/ops/conv2d.h>
#include <ir/ops/conv2d_transpose.h>
#include <ir/ops/dequantize.h>
#include <ir/ops/fake_dequantize.h>
#include <ir/ops/fake_quantize.h>
#include <ir/ops/matmul.h>
#include <ir/ops/pad.h>
#include <ir/ops/quantize.h>
#include <ir/ops/reduce.h>
#include <ir/ops/reduce_window2d.h>
#include <ir/ops/reshape.h>
#include <ir/ops/resize_image.h>
#include <ir/ops/strided_slice.h>
#include <ir/ops/transpose.h>
#include <ir/ops/unary.h>
#include <runtime/neutral/neutral_ops_body.h>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::runtime;
using namespace nncase::ir;
using namespace nncase::runtime::neutral;

namespace nncase
{
namespace codegen
{
    void register_netural_emitters()
    {
        disable_emitter(op_input_node);
        disable_emitter(op_output_node);
        disable_emitter(op_ignore_node);
        disable_emitter(op_constant);

        register_emitter(op_binary, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<binary &>(node);
            auto body = std::make_unique<node_body_impl<rop_binary, binary_options>>();

            body->input_a = context.get_allocation(rnode.input_a());
            body->input_b = context.get_allocation(rnode.input_b());
            body->output = context.get_allocation(rnode.output());
            body->binary_op = rnode.binary_op();
            body->in_a_shape = to(rnode.input_a().shape());
            body->in_b_shape = to(rnode.input_b().shape());
            body->out_shape = to(rnode.output().shape());
            body->fused_activation = rnode.fused_activation();

            return body;
        });

        register_emitter(op_quantized_binary, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<quantized_binary &>(node);
            auto body = std::make_unique<node_body_impl<rop_quantized_binary, quantized_binary_options>>();

            body->input_a = context.get_allocation(rnode.input_a());
            body->input_b = context.get_allocation(rnode.input_b());
            body->output = context.get_allocation(rnode.output());
            body->binary_op = rnode.binary_op();
            body->in_a_shape = to(rnode.input_a().shape());
            body->in_b_shape = to(rnode.input_b().shape());
            body->out_shape = to(rnode.output().shape());
            body->input_a_offset = rnode.input_a_offset();
            body->input_a_mul = rnode.input_a_mul();
            body->input_a_shift = rnode.input_a_shift();
            body->input_b_offset = rnode.input_b_offset();
            body->input_b_mul = rnode.input_b_mul();
            body->input_b_shift = rnode.input_b_shift();
            body->output_offset = rnode.output_offset();
            body->output_mul = rnode.output_mul();
            body->output_shift = rnode.output_shift();

            return body;
        });

        register_emitter(op_concat, [](node &node, codegen_context &context) {
            struct concat_options_body : public node_body_impl<rop_concat, concat_options>
            {
                std::vector<memory_range> inputs_holder;
            };

            auto &rnode = static_cast<concat &>(node);
            auto body = std::make_unique<concat_options_body>();

            for (auto &&in : rnode.inputs())
                body->inputs_holder.emplace_back(context.get_allocation(in));

            auto elem_size = (uint32_t)runtime::get_bytes(rnode.output().type());
            uint64_t inner_size, outer_size;
            get_concat_params(rnode.output().shape(), elem_size, rnode.axis(), inner_size, outer_size);

            body->output = context.get_allocation(rnode.output());
            body->inner_size = inner_size;
            body->outer_size = outer_size;
            body->inputs_count = (uint32_t)body->inputs_holder.size();
            body->inputs = body->inputs_holder;
            body->dims = rnode.concat_dims();

            return body;
        });

        register_emitter(op_conv2d, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<conv2d &>(node);
            auto body = std::make_unique<node_body_impl<rop_conv2d, conv2d_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());
            body->groups = rnode.groups();
            body->out_channels = rnode.output_channels();
            body->padding_h = rnode.padding_h();
            body->padding_w = rnode.padding_w();
            body->filter_h = rnode.filter_h();
            body->filter_w = rnode.filter_w();
            body->stride_h = rnode.stride_h();
            body->stride_w = rnode.stride_w();
            body->dilation_h = rnode.dilation_h();
            body->dilation_w = rnode.dilation_w();
            body->fused_activation = rnode.fused_activation();
            body->weights = rnode.weights();
            body->bias = rnode.bias();

            return body;
        });

        register_emitter(op_quantized_conv2d, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<quantized_conv2d &>(node);
            auto body = std::make_unique<node_body_impl<rop_quantized_conv2d, quantized_conv2d_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());
            body->groups = rnode.groups();
            body->out_channels = rnode.output_channels();
            body->padding_h = rnode.padding_h();
            body->padding_w = rnode.padding_w();
            body->filter_h = rnode.filter_h();
            body->filter_w = rnode.filter_w();
            body->stride_h = rnode.stride_h();
            body->stride_w = rnode.stride_w();
            body->dilation_h = rnode.dilation_h();
            body->dilation_w = rnode.dilation_w();
            body->input_offset = rnode.input_offset();
            body->filter_offset = rnode.filter_offset();
            body->output_mul = rnode.output_mul();
            body->output_shift = rnode.output_shift();
            body->output_offset = rnode.output_offset();
            body->weights = rnode.weights();
            body->bias = rnode.bias();

            return body;
        });

        register_emitter(op_conv2d_transpose, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<conv2d_transpose &>(node);
            auto body = std::make_unique<node_body_impl<rop_conv2d_transpose, conv2d_transpose_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());
            body->out_shape = to(rnode.output().shape());
            body->groups = rnode.groups();
            body->padding_h = rnode.padding_h();
            body->padding_w = rnode.padding_w();
            body->filter_h = rnode.filter_h();
            body->filter_w = rnode.filter_w();
            body->stride_h = rnode.stride_h();
            body->stride_w = rnode.stride_w();
            body->dilation_h = rnode.dilation_h();
            body->dilation_w = rnode.dilation_w();
            body->fused_activation = rnode.fused_activation();
            body->weights = rnode.weights();
            body->bias = rnode.bias();

            return body;
        });

        register_emitter(op_dequantize, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<dequantize &>(node);
            auto body = std::make_unique<node_body_impl<rop_dequantize, dequantize_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->quant_param = rnode.quant_param();

            return body;
        });

        register_emitter(op_matmul, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<matmul &>(node);
            auto body = std::make_unique<node_body_impl<rop_matmul, matmul_options>>();

            body->input_a = context.get_allocation(rnode.input_a());
            body->input_b = context.get_allocation(rnode.input_b());
            body->output = context.get_allocation(rnode.output());
            body->a_rows = rnode.input_a().shape()[0];
            body->a_cols = rnode.input_a().shape()[1];
            body->b_cols = rnode.input_b().shape()[1];
            body->fused_activation = rnode.fused_activation();
            body->bias = rnode.bias();

            return body;
        });

        register_emitter(op_quantized_matmul, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<quantized_matmul &>(node);
            auto body = std::make_unique<node_body_impl<rop_quantized_matmul, quantized_matmul_options>>();

            body->input_a = context.get_allocation(rnode.input_a());
            body->input_b = context.get_allocation(rnode.input_b());
            body->output = context.get_allocation(rnode.output());
            body->a_rows = rnode.input_a().shape()[0];
            body->a_cols = rnode.input_a().shape()[1];
            body->b_cols = rnode.input_b().shape()[1];
            body->input_a_offset = rnode.input_a_offset();
            body->input_b_offset = rnode.input_b_offset();
            body->output_mul = rnode.output_mul();
            body->output_shift = rnode.output_shift();
            body->output_offset = rnode.output_offset();
            body->bias = rnode.bias();

            return body;
        });

        register_emitter(op_pad, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<pad &>(node);
            auto body = std::make_unique<node_body_impl<rop_pad, pad_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());
            body->paddings = to(rnode.paddings());

            return body;
        });

        register_emitter(op_quantize, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<quantize &>(node);
            auto body = std::make_unique<node_body_impl<rop_quantize, quantize_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->quant_param = rnode.quant_param();

            return body;
        });

        register_emitter(op_reduce, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<reduce &>(node);
            auto body = std::make_unique<node_body_impl<rop_reduce, reduce_options>>();

            auto reduced_shape = get_reduced_shape(rnode.input().shape(), rnode.axis(), true);

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->reduce_op = rnode.reduce_op();
            body->in_shape = to(rnode.input().shape());
            body->out_shape = to(reduced_shape);
            body->init_value = rnode.init_value();

            return body;
        });

        register_emitter(op_reduce_window2d, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<reduce_window2d &>(node);
            auto body = std::make_unique<node_body_impl<rop_reduce_window2d, reduce_window2d_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->reduce_op = rnode.reduce_op();
            body->in_shape = to(rnode.input().shape());
            body->padding_h = rnode.padding_h();
            body->padding_w = rnode.padding_w();
            body->filter_h = rnode.filter_h();
            body->filter_w = rnode.filter_w();
            body->stride_h = rnode.stride_h();
            body->stride_w = rnode.stride_w();
            body->dilation_h = rnode.dilation_h();
            body->dilation_w = rnode.dilation_w();
            body->init_value = rnode.init_value();
            body->fused_activation = rnode.fused_activation();

            return body;
        });

        register_emitter(op_reshape, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<reshape &>(node);
            auto body = std::make_unique<node_body_impl<rop_memory_copy, memory_copy_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());

            return body;
        });

        register_emitter(op_resize_image, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<resize_image &>(node);
            auto body = std::make_unique<node_body_impl<rop_resize_image, resize_image_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());
            body->out_h = rnode.new_size()[0];
            body->out_w = rnode.new_size()[1];
            body->mode = rnode.mode();
            body->align_corners = rnode.align_corners();

            return body;
        });

        register_emitter(op_strided_slice, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<strided_slice &>(node);
            auto body = std::make_unique<node_body_impl<rop_strided_slice, strided_slice_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());
            body->begin = to(rnode.begin());
            body->end = to(rnode.end());
            body->strides = to(rnode.strides());
            body->begin_mask = rnode.begin_mask();
            body->end_mask = rnode.end_mask();
            body->ellipsis_mask = rnode.ellipsis_mask();
            body->new_axis_mask = rnode.new_axis_mask();
            body->shrink_axis_mask = rnode.shrink_axis_mask();

            return body;
        });

        register_emitter(op_transpose, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<transpose &>(node);
            auto body = std::make_unique<node_body_impl<rop_transpose, transpose_options>>();

            runtime_shape_t in_shape, perm;
            extend_transpose_shape(rnode.input().shape(), rnode.perm(), in_shape, perm);

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = in_shape;
            body->perm = perm;

            return body;
        });

        register_emitter(op_unary, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<unary &>(node);
            auto body = std::make_unique<node_body_impl<rop_unary, unary_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->unary_op = rnode.unary_op();

            return body;
        });
    }
}
}
