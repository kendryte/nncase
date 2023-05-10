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
#include <nncase/codegen/nnil_builder.h>
#include <nncase/ir/evaluator.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/batch_to_space.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/broadcast.h>
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/compare.h>
#include <nncase/ir/ops/compress.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/conv2d_transpose.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/cumsum.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/fused_unary.h>
#include <nncase/ir/ops/gather.h>
#include <nncase/ir/ops/gather_elements.h>
#include <nncase/ir/ops/gather_nd.h>
#include <nncase/ir/ops/gru.h>
#include <nncase/ir/ops/hardmax.h>
#include <nncase/ir/ops/instancenorm.h>
#include <nncase/ir/ops/layernorm.h>
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/ops/onehot.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/random_normal.h>
#include <nncase/ir/ops/random_uniform.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/reduce_arg.h>
#include <nncase/ir/ops/reduce_prod.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/ops/resize_image.h>
#include <nncase/ir/ops/roi_align.h>
#include <nncase/ir/ops/sigmoid.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/softmax.h>
#include <nncase/ir/ops/space_to_batch.h>
#include <nncase/ir/ops/table_lookup.h>
#include <nncase/ir/ops/ternary.h>
#include <nncase/ir/ops/tflite_detection_postprocess.h>
#include <nncase/ir/ops/topk.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/ops/trilu.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/runtime_type_utils.h>
#include <nncase/kernels/convolution.h>
#include <nncase/kernels/neutral/neutral_kernels.h>
#include <nncase/kernels/nnil.h>
#include <nncase/kernels/reduce_window.h>
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/debug.h>

using namespace nncase;
using namespace nncase::schedule;
using namespace nncase::ir;
using namespace nncase::kernels;
using namespace nncase::runtime;

#define ELEM_SIZE_IMPL(type, KERNEL)                            \
    switch (get_bytes(type))                                    \
    {                                                           \
    case 1:                                                     \
        KERNEL(uint8_t);                                        \
        break;                                                  \
    case 2:                                                     \
        KERNEL(uint16_t);                                       \
        break;                                                  \
    case 4:                                                     \
        KERNEL(uint32_t);                                       \
        break;                                                  \
    default:                                                    \
        throw std::runtime_error("Not supported element type"); \
    }

namespace
{
void nop_evaluator(ir::node &, function_evaluate_context &)
{
}
}

namespace nncase::ir
{
void register_neutral_evaluators()
{
    register_evaluator(op_input_node, nop_evaluator);
    register_evaluator(op_output_node, nop_evaluator);
    register_evaluator(op_ignore_node, nop_evaluator);
    register_evaluator(op_constant, nop_evaluator);

    register_evaluator(op_batch_to_space, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<batch_to_space &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());

        kernels::batch_to_space(input.datatype(), input.buffer().data(), output.buffer().data(), input.shape(),
            runtime_shape_t { (size_t)rnode.block_size_h(), (size_t)rnode.block_size_w() },
            runtime_paddings_t { padding { rnode.crop_h()[0], rnode.crop_h()[1] }, padding { rnode.crop_w()[0], rnode.crop_w()[1] } },
            input.strides(), output.strides())
            .unwrap_or_throw(); });

    register_evaluator(op_binary, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<binary &>(node);

        auto input_a = context.memory_at(rnode.input_a());
        auto input_b = context.memory_at(rnode.input_b());
        auto output = context.memory_at(rnode.output());

        auto input_type = rnode.input_a().type();
        switch (input_type)
        {
        case dt_float32:
            kernels::binary(rnode.binary_op(), input_a.buffer().as_span<float>().data(), input_b.buffer().as_span<float>().data(),
                output.buffer().as_span<float>().data(), input_a.shape(), input_a.strides(), input_b.shape(), input_b.strides(), output.shape(), output.strides(),
                rnode.fused_activation())
                .unwrap_or_throw();
            break;
        case dt_int32:
            kernels::binary(rnode.binary_op(), input_a.buffer().as_span<int32_t>().data(), input_b.buffer().as_span<int32_t>().data(),
                output.buffer().as_span<int32_t>().data(), input_a.shape(), input_a.strides(), input_b.shape(), input_b.strides(), output.shape(), output.strides(),
                rnode.fused_activation())
                .unwrap_or_throw();
            break;
        case dt_int64:
            kernels::binary(rnode.binary_op(), input_a.buffer().as_span<int64_t>().data(), input_b.buffer().as_span<int64_t>().data(),
                output.buffer().as_span<int64_t>().data(), input_a.shape(), input_a.strides(), input_b.shape(), input_b.strides(), output.shape(), output.strides(),
                rnode.fused_activation())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for binary: " + std::string(datatype_names(input_type));
        } });

    register_evaluator(op_broadcast, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<broadcast &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        kernels::broadcast(input.datatype(), input.buffer().data(), output.buffer().data(),
            input.shape(), input.strides(), output.shape(), output.strides())
            .unwrap_or_throw(); });

    register_evaluator(op_concat, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<concat &>(node);

        std::vector<const gsl::byte *> inputs_mem;
        std::vector<runtime_shape_t> inputs_strides;
        for (auto in : rnode.inputs())
        {
            auto input = context.memory_at(*in);
            inputs_mem.emplace_back(input.buffer().data());
            inputs_strides.emplace_back(input.strides());
        }

        auto output = context.memory_at(rnode.output());
        runtime_shape_t concat_dims { rnode.concat_dims().begin(), rnode.concat_dims().end() };
        kernels::concat(rnode.output().type(), inputs_mem, output.buffer().data(), output.shape(), inputs_strides,
            output.strides(), rnode.axis(), concat_dims)
            .unwrap_or_throw(); });

    register_evaluator(op_conv2d, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<conv2d &>(node);

        assert(rnode.input().type() == dt_float32);

        auto input = context.memory_at(rnode.input());
        auto weights = context.memory_at(rnode.weights());
        auto bias = context.memory_at(rnode.bias());
        auto output = context.memory_at(rnode.output());
        auto input_mem = input.buffer().as_span<float>();
        auto weights_mem = weights.buffer().as_span<float>();
        auto bias_mem = bias.buffer().as_span<float>();
        auto output_mem = output.buffer().as_span<float>();

        kernels::conv2d(input_mem.data(), weights_mem.data(), bias_mem.data(), output_mem.data(), input.shape(), input.strides(),
            weights.shape(), weights.strides(), bias.strides(), output.strides(), rnode.padding_h(), rnode.padding_w(),
            rnode.groups(), rnode.stride_h(), rnode.stride_w(), rnode.dilation_h(), rnode.dilation_w(), rnode.fused_activation())
            .unwrap_or_throw(); });

    register_evaluator(op_conv2d_transpose, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<conv2d_transpose &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at(rnode.input()).buffer().as_span<float>();
        auto weights = context.memory_at(rnode.weights()).buffer().as_span<float>();
        auto bias = context.memory_at(rnode.bias()).buffer().as_span<float>();
        auto output = context.memory_at(rnode.output()).buffer().as_span<float>();

        neutral::conv2d_transpose(input.data(), output.data(), weights.data(), bias.data(), to(rnode.input().shape()),
            rnode.groups(), to(rnode.output().shape()), rnode.filter_h(), rnode.filter_w(), rnode.stride_h(), rnode.stride_w(),
            rnode.dilation_h(), rnode.dilation_w(), rnode.padding_h(), rnode.padding_w(), rnode.fused_activation()); });

    register_evaluator(op_dequantize, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<dequantize &>(node);

        auto output = context.memory_at(rnode.output()).buffer().as_span<float>();

        switch (rnode.input().type())
        {
#define DEQUANTIZE(type)                                                                                                \
    case type:                                                                                                          \
    {                                                                                                                   \
        auto input = context.memory_at(rnode.input()).buffer().as_span<to_cpp_type_t<type>>();                          \
        neutral::dequantize(input.data(), output.data(), xt::compute_size(rnode.input().shape()), rnode.quant_param()); \
        break;                                                                                                          \
    }
            DEQUANTIZE(dt_uint8)
            DEQUANTIZE(dt_int8)
            DEQUANTIZE(dt_int16)
            DEQUANTIZE(dt_int32)
        default:
            assert(false && "not supported type!");

#undef DEQUANTIZE
        } });

    register_evaluator(op_compare, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<compare &>(node);

        auto input_a = context.memory_at(rnode.input_a());
        auto input_b = context.memory_at(rnode.input_b());
        auto output = context.memory_at(rnode.output());

        auto input_type = rnode.input_a().type();
        switch (input_type)
        {
        case dt_uint8:
            kernels::compare(rnode.compare_op(), input_a.buffer().as_span<uint8_t>().data(), input_b.buffer().as_span<uint8_t>().data(),
                output.buffer().as_span<bool>().data(), input_a.shape(), input_a.strides(),
                input_b.shape(), input_b.strides(), output.shape(), output.strides())
                .unwrap_or_throw();
            break;
        case dt_float32:
            kernels::compare(rnode.compare_op(), input_a.buffer().as_span<float>().data(), input_b.buffer().as_span<float>().data(),
                output.buffer().as_span<bool>().data(), input_a.shape(), input_a.strides(),
                input_b.shape(), input_b.strides(), output.shape(), output.strides())
                .unwrap_or_throw();
            break;
        case dt_int32:
            kernels::compare(rnode.compare_op(), input_a.buffer().as_span<int32_t>().data(), input_b.buffer().as_span<int32_t>().data(),
                output.buffer().as_span<bool>().data(), input_a.shape(), input_a.strides(),
                input_b.shape(), input_b.strides(), output.shape(), output.strides())
                .unwrap_or_throw();
            break;
        case dt_int64:
            kernels::compare(rnode.compare_op(), input_a.buffer().as_span<int64_t>().data(), input_b.buffer().as_span<int64_t>().data(),
                output.buffer().as_span<bool>().data(), input_a.shape(), input_a.strides(),
                input_b.shape(), input_b.strides(), output.shape(), output.strides())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for compare: " + std::string(datatype_names(input_type));
        } });

    register_evaluator(op_fused_unary, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<fused_unary &>(node);

        auto input = context.memory_at(rnode.input()).buffer().as_span<float>();
        auto output = context.memory_at(rnode.output()).buffer().as_span<float>();

        using namespace nncase::codegen;
        std::stringstream ss;
        binary_writer bw(ss);
        nnil_builder builder(bw);

        fused_unary::compile_graph(rnode.subgraph(), builder);
        auto buf = ss.str();
        std::vector<gsl::byte> body(reinterpret_cast<gsl::byte *>(buf.data()), reinterpret_cast<gsl::byte *>(buf.data() + buf.size()));
        kernels::nnil_unary_method(input.data(), output.data(), input.size(), body)
            .unwrap_or_throw(); });

    register_evaluator(op_matmul, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<matmul &>(node);

        assert(rnode.input_a().type() == dt_float32);
        assert(rnode.input_b().type() == dt_float32);
        auto input_a = context.memory_at(rnode.input_a());
        auto input_b = context.memory_at(rnode.input_b());
        auto bias = context.memory_at(rnode.bias());
        auto output = context.memory_at(rnode.output());
        auto input_a_mem = input_a.buffer().as_span<float>();
        auto input_b_mem = input_b.buffer().as_span<float>();
        auto bias_mem = bias.buffer().as_span<float>();
        auto output_mem = output.buffer().as_span<float>();

        kernels::matmul(input_a_mem.data(), input_b_mem.data(), bias_mem.data(), output_mem.data(), input_a.shape(), input_a.strides(),
            input_b.shape(), input_b.strides(), output.shape(), output.strides(), rnode.fused_activation())
            .unwrap_or_throw(); });

    register_evaluator(op_pad, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<pad &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        auto input_mem = input.buffer();
        auto output_mem = output.buffer();

        kernels::pad(input.datatype(), input_mem.data(), output_mem.data(), input.shape(), input.strides(),
            output.strides(), to(rnode.paddings()), rnode.pad_mode(), rnode.pad_value())
            .unwrap_or_throw(); });

    register_evaluator(op_quantize, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<quantize &>(node);
        auto input = context.memory_at(rnode.input()).buffer().as_span<float>();
        switch (rnode.output().type())
        {
#define QUANTIZE(type)                                                                                                \
    case type:                                                                                                        \
    {                                                                                                                 \
        auto output = context.memory_at(rnode.output()).buffer().as_span<to_cpp_type_t<type>>();                      \
        neutral::quantize(input.data(), output.data(), xt::compute_size(rnode.input().shape()), rnode.quant_param()); \
        break;                                                                                                        \
    }
            QUANTIZE(dt_uint8)
            QUANTIZE(dt_int8)
            QUANTIZE(dt_int16)
        default:
            assert(false && "not supported type!");
#undef QUANTIZE
        } });

    register_evaluator(op_reduce, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<reduce &>(node);
        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());

        auto input_type = rnode.input().type();
        switch (input_type)
        {
        case dt_float32:
            kernels::reduce(rnode.reduce_op(), static_cast<float>(rnode.init_value()), input.buffer().as_span<float>().data(),
                output.buffer().as_span<float>().data(), input.shape(), to(rnode.axis()), input.strides(), output.strides(), rnode.keep_dims())
                .unwrap_or_throw();
            break;
        case dt_int32:
            kernels::reduce(rnode.reduce_op(), static_cast<int32_t>(rnode.init_value()), input.buffer().as_span<int32_t>().data(),
                output.buffer().as_span<int32_t>().data(), input.shape(), to(rnode.axis()), input.strides(), output.strides(), rnode.keep_dims())
                .unwrap_or_throw();
            break;
        case dt_int64:
            kernels::reduce(rnode.reduce_op(), static_cast<int64_t>(rnode.init_value()), input.buffer().as_span<int64_t>().data(),
                            output.buffer().as_span<int64_t>().data(), input.shape(), to(rnode.axis()), input.strides(), output.strides(), rnode.keep_dims())
                    .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for reduce: " + std::string(datatype_names(input_type));
        } });

    register_evaluator(op_reduce_arg, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<reduce_arg &>(node);
        assert(rnode.input().type() == dt_float32);
        auto output_type = rnode.output().type();
        assert(output_type == dt_int32 || output_type == dt_int64);
        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        auto input_mem = input.buffer().as_span<float>();
        axis_t axes { rnode.axis() };

        switch (output_type)
        {
        case dt_int32:
            kernels::reduce_arg(rnode.reduce_arg_op(), input_mem.data(), output.buffer().as_span<int32_t>().data(), input.shape(),
                input.strides(), output.strides(), to(axes), rnode.keep_dims(), rnode.select_last_index())
                .unwrap_or_throw();
            break;
        case dt_int64:
            kernels::reduce_arg(rnode.reduce_arg_op(), input_mem.data(), output.buffer().as_span<int64_t>().data(), input.shape(),
                input.strides(), output.strides(), to(axes), rnode.keep_dims(), rnode.select_last_index())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for reduce_arg: " + std::string(datatype_names(output_type));
        } });

    register_evaluator(op_reduce_prod, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<reduce_prod &>(node);
        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());

        auto input_type = rnode.input().type();
        switch (input_type)
        {
        case dt_float32:
            kernels::reduce_prod(input.buffer().as_span<float>().data(), output.buffer().as_span<float>().data(), input.shape(),
                input.strides(), output.strides(), to(rnode.axis()), rnode.keep_dims())
                .unwrap_or_throw();
            break;
        case dt_int32:
            kernels::reduce_prod(input.buffer().as_span<int32_t>().data(), output.buffer().as_span<int32_t>().data(), input.shape(),
                input.strides(), output.strides(), to(rnode.axis()), rnode.keep_dims())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for reduce_prod: " + std::string(datatype_names(input_type));
        } });

    register_evaluator(op_reduce_window2d, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<reduce_window2d &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        auto input_mem = input.buffer().as_span<float>();
        auto output_mem = output.buffer().as_span<float>();

        kernels::reduce_window2d(rnode.reduce_op(), input_mem.data(), rnode.init_value(), output_mem.data(),
            input.shape(), input.strides(), output.strides(), rnode.padding_h(), rnode.padding_w(), rnode.filter_h(), rnode.filter_w(),
            rnode.stride_h(), rnode.stride_w(), rnode.dilation_h(), rnode.dilation_w(), rnode.fused_activation())
            .unwrap_or_throw(); });

    register_evaluator(op_bitcast, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<bitcast &>(node);

        auto input = context.memory_at(rnode.input()).buffer();
        auto output = context.memory_at(rnode.output()).buffer();

        std::copy(input.begin(), input.end(), output.begin()); });

    register_evaluator(op_resize_image, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<resize_image &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        auto input_data = input.buffer().data();
        auto output_data = output.buffer().data();
        auto new_size = rnode.new_size();
        if (rnode.mode() == image_resize_bilinear)
        {
            kernels::resize_bilinear(input.datatype(), input_data, output_data,
                input.shape(), input.strides(), output.strides(), new_size[0], new_size[1],
                rnode.align_corners(), rnode.half_pixel_centers())
                .unwrap_or_throw();
        }
        else
        {
            kernels::resize_nearest_neighbor(input.datatype(), input_data, output_data,
                input.shape(), input.strides(), output.strides(), new_size[0], new_size[1],
                rnode.align_corners(), rnode.half_pixel_centers())
                .unwrap_or_throw();
        } });

    register_evaluator(op_roi_align, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<roi_align &>(node);

        auto input = context.memory_at(rnode.input());
        auto rois = context.memory_at(rnode.rois());
        auto batch_indices = context.memory_at(rnode.batch_indices());
        auto output = context.memory_at(rnode.output());

        auto input_type = rnode.input().type();
        switch (input_type)
        {
        case dt_float32:
            kernels::roi_align(input.buffer().as_span<float>().data(), rois.buffer().as_span<float>().data(),
                batch_indices.buffer().as_span<int64_t>().data(), output.buffer().as_span<float>().data(), input.shape(), output.shape(),
                rnode.mode(), rnode.spatial_scale(), rnode.sampling_ratio())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for roi_align: " + std::string(datatype_names(input_type));
        } });

    register_evaluator(op_sigmoid, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<sigmoid &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());

        auto output_type = rnode.output().type();
        switch (output_type)
        {
        case dt_float32:
            kernels::sigmoid(input.buffer().as_span<float>().data(), output.buffer().as_span<float>().data(), input.shape(),
                input.strides(), output.strides())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for sigmoid: " + std::string(datatype_names(output_type));
        } });

    register_evaluator(op_slice, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<slice &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        auto input_mem = input.buffer();
        auto output_mem = output.buffer();

        kernels::slice(input.datatype(), input_mem.data(), output_mem.data(), input.shape(),
            input.strides(), output.strides(), to(rnode.begin()), to<int32_t>(rnode.end()), to<int32_t>(rnode.strides()))
            .unwrap_or_throw(); });

    register_evaluator(op_softmax, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<softmax &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());

        auto output_type = rnode.output().type();
        switch (output_type)
        {
        case dt_float32:
            kernels::softmax(input.buffer().as_span<float>().data(), output.buffer().as_span<float>().data(), input.shape(),
                input.strides(), output.strides(), rnode.axis(), rnode.beta())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for softmax: " + std::string(datatype_names(output_type));
        } });

    register_evaluator(op_space_to_batch, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<space_to_batch &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());

        kernels::space_to_batch(input.datatype(), input.buffer().data(), output.buffer().data(), input.shape(),
            runtime_shape_t { (size_t)rnode.block_size_h(), (size_t)rnode.block_size_w() },
            runtime_paddings_t { rnode.padding_h(), rnode.padding_w() },
            input.strides(), output.strides())
            .unwrap_or_throw(); });

    register_evaluator(op_ternary, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<ternary &>(node);

        auto input_a = context.memory_at(rnode.input_a());
        auto input_b = context.memory_at(rnode.input_b());
        auto input_c = context.memory_at(rnode.input_c());
        auto output = context.memory_at(rnode.output());

        auto output_type = rnode.output().type();
        switch (output_type)
        {
        case dt_float32:
            kernels::ternary(input_a.buffer().as_span<float>().data(), input_b.buffer().as_span<float>().data(),
                input_c.buffer().as_span<float>().data(), output.buffer().as_span<float>().data(), input_a.shape(), input_a.strides(),
                input_b.shape(), input_b.strides(), input_c.shape(), input_c.strides(), output.strides())
                .unwrap_or_throw();
            break;
        case dt_int64:
            kernels::ternary(input_a.buffer().as_span<float>().data(), input_b.buffer().as_span<int64_t>().data(),
                input_c.buffer().as_span<int64_t>().data(), output.buffer().as_span<int64_t>().data(), input_a.shape(), input_a.strides(),
                input_b.shape(), input_b.strides(), input_c.shape(), input_c.strides(), output.strides())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for ternary: " + std::string(datatype_names(output_type));
        } });

    register_evaluator(op_transpose, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<transpose &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        auto input_mem = input.buffer();
        auto output_mem = output.buffer();

        kernels::transpose(input.datatype(), input_mem.data(), output_mem.data(), input.shape(), to(rnode.perm()),
            input.strides(), output.strides())
            .unwrap_or_throw(); });

    register_evaluator(op_unary, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<unary &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at(rnode.input()).buffer().as_span<float>();
        auto output = context.memory_at(rnode.output()).buffer().as_span<float>();

        auto unary = [&](auto unary_op) {
            neutral::unary(input.data(), output.data(), input.size(), unary_op);
        };

        switch (rnode.unary_op())
        {
        case unary_abs:
            unary([](auto a) { return fabs(a); });
            break;
        case unary_acos:
            unary([](auto a) { return acosf(a); });
            break;
        case unary_asin:
            unary([](auto a) { return asinf(a); });
            break;
        case unary_ceil:
            unary([](auto a) { return ceilf(a); });
            break;
        case unary_cos:
            unary([](auto a) { return cosf(a); });
            break;
        case unary_exp:
            unary([](auto a) { return expf(a); });
            break;
        case unary_floor:
            unary([](auto a) { return floorf(a); });
            break;
        case unary_log:
            unary([](auto a) { return logf(a); });
            break;
        case unary_logical_not:
            unary([](auto a) { return !a; });
            break;
        case unary_neg:
            unary([](auto a) { return -a; });
            break;
        case unary_round:
            unary([](auto a) { return rintf(a); });
            break;
        case unary_rsqrt:
            unary([](auto a) { return 1.f / sqrtf(a); });
            break;
        case unary_sign:
            unary([](auto a) { return (0 < a) - (a < 0); });
            break;
        case unary_sin:
            unary([](auto a) { return sinf(a); });
            break;
        case unary_sqrt:
            unary([](auto a) { return sqrt(a); });
            break;
        case unary_square:
            unary([](auto a) { return a * a; });
            break;
        case unary_tanh:
            unary([](auto a) { return tanh(a); });
            break;
        case unary_erf:
            unary([](auto a) { return erf(a); });
            break;
        default:
            throw std::runtime_error("Not supported unary");
        } });

    register_evaluator(op_table_lookup1d, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<table_lookup1d &>(node);

        assert(rnode.input().type() == dt_uint8);
        auto input = context.memory_at(rnode.input()).buffer().as_span<uint8_t>();
        auto table = context.memory_at(rnode.table()).buffer().as_span<uint8_t>();
        auto output = context.memory_at(rnode.output()).buffer().as_span<uint8_t>();

        kernels::neutral::table_lookup1d(input.data(), output.data(), input.size(), table.data()); });

    register_evaluator(op_clamp, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<clamp &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at(rnode.input()).buffer().as_span<float>();
        auto input_low = context.memory_at(rnode.input_low()).buffer().as_span<float>();
        auto input_high = context.memory_at(rnode.input_high()).buffer().as_span<float>();
        auto output = context.memory_at(rnode.output()).buffer().as_span<float>();

        const float *input_ptr = input.data();
        float low = input_low.data()[0];
        float high = input_high.data()[0];
        float *output_ptr = output.data();
        for (size_t i = 0; i < input.size(); i++)
        {
            output_ptr[i] = std::clamp(input_ptr[i], low, high);
        } });

    register_evaluator(op_convert, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<convert &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        auto input_mem = input.buffer();
        auto output_mem = output.buffer();

        kernels::convert(input.datatype(), output.datatype(), input_mem.data(), output_mem.data(), input.shape(),
            input.strides(), output.strides())
            .unwrap_or_throw(); });

    register_evaluator(op_gather, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<gather &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        auto indices = context.memory_at(rnode.indices());
        auto input_mem = input.buffer();
        auto output_mem = output.buffer();

        kernels::gather(input.datatype(), input_mem.data(), output_mem.data(), input.shape(), output.shape(),
            input.strides(), output.strides(), reinterpret_cast<const int32_t *>(indices.buffer().data()), indices.shape(), rnode.axis())
            .unwrap_or_throw(); });

    register_evaluator(op_gather_nd, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<gather_nd &>(node);

        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());
        auto indices = context.memory_at(rnode.indices());
        auto input_mem = input.buffer();
        auto output_mem = output.buffer();

        kernels::gather_nd(input.datatype(), input_mem.data(), output_mem.data(), input.shape(), output.shape(),
            input.strides(), output.strides(), reinterpret_cast<const int32_t *>(indices.buffer().data()), indices.shape(), rnode.batch_dims())
            .unwrap_or_throw(); });

    register_evaluator(op_onehot, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<onehot &>(node);

        auto indices = context.memory_at(rnode.indices());
        auto depth = context.memory_at(rnode.depth());
        auto on_value = context.memory_at(rnode.on_value());
        auto off_value = context.memory_at(rnode.off_value());
        auto output = context.memory_at(rnode.output());
        auto indices_mem = reinterpret_cast<const int32_t *>(indices.buffer().data());
        auto output_mem = output.buffer().data();
        auto depth_mem = depth.buffer().data();
        auto on_value_mem = on_value.buffer().data();
        auto off_value_mem = off_value.buffer().data();
        kernels::onehot(output.datatype(), indices_mem, output_mem, indices.shape(), output.shape(),
            output.strides(), depth_mem, off_value_mem, on_value_mem, rnode.axis(), rnode.mode())
            .unwrap_or_throw(); });

    register_evaluator(op_cumsum, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<cumsum &>(node);
        auto datatype = rnode.input().type();
        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());

        switch (datatype)
        {
        case dt_float32:
            kernels::cumsum(input.buffer().as_span<float>().data(), output.buffer().as_span<float>().data(),
                input.shape(), rnode.axis(), rnode.exclusive(), rnode.reverse())
                .unwrap_or_throw();
            break;
        case dt_int32:
            kernels::cumsum(input.buffer().as_span<int32_t>().data(), output.buffer().as_span<int32_t>().data(),
                input.shape(), rnode.axis(), rnode.exclusive(), rnode.reverse())
                .unwrap_or_throw();
            break;
        default:
            throw std::runtime_error("unsupported dtype for cumsum: " + std::string(datatype_names(datatype)));
        } });

    register_evaluator(op_hardmax, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<hardmax &>(node);
        auto datatype = rnode.input().type();
        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());

        switch (datatype)
        {
        case dt_float32:
            kernels::hardmax(input.buffer().as_span<float>().data(), input.shape(), input.strides(),
                output.buffer().as_span<float>().data(), rnode.axis())
                .unwrap_or_throw();
            break;
        default:
            throw std::runtime_error("unsupported dtype for hardmax: " + std::string(datatype_names(datatype)));
        } });

    register_evaluator(op_random_normal, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<random_normal &>(node);
        auto datatype = rnode.output().type();
        auto output = context.memory_at(rnode.output());

        switch (datatype)
        {
        case dt_float32:
            kernels::random_normal(output.buffer().as_span<float>().data(), output.shape(), rnode.mean(), rnode.std(), rnode.seed())
                .unwrap_or_throw();
            break;
        default:
            throw std::runtime_error("unsupported dtype for random_normal: " + std::string(datatype_names(datatype)));
        } });

    register_evaluator(op_random_uniform, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<random_uniform &>(node);
        auto datatype = rnode.output().type();
        auto output = context.memory_at(rnode.output());

        switch (datatype)
        {
        case dt_float32:
            kernels::random_uniform(output.buffer().as_span<float>().data(), output.shape(), rnode.low(), rnode.high(), rnode.seed())
                .unwrap_or_throw();
            break;
        default:
            throw std::runtime_error("unsupported dtype for random_uniform: " + std::string(datatype_names(datatype)));
        } });

    register_evaluator(op_topk, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<topk &>(node);
        auto datatype = rnode.input().type();
        auto input = context.memory_at(rnode.input());
        auto output_values = context.memory_at(rnode.output_a());
        auto output_indices = context.memory_at(rnode.output_b());

        switch (datatype)
        {
        case dt_float32:
            kernels::topk(input.buffer().as_span<float>().data(), output_values.buffer().as_span<float>().data(),
                output_indices.buffer().as_span<int64_t>().data(),
                input.shape(), input.strides(), output_values.shape(), output_values.strides(),
                output_indices.shape(), output_indices.strides(),
                rnode.k(), rnode.axis(), rnode.largest(), rnode.sorted())
                .unwrap_or_throw();
            break;
        default:
            throw std::runtime_error("unsupported dtype for topk: " + std::string(datatype_names(datatype)));
        } });

    register_evaluator(op_trilu, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<trilu &>(node);
        auto datatype = rnode.input().type();
        auto input = context.memory_at(rnode.input());
        auto output = context.memory_at(rnode.output());

        switch (datatype)
        {
        case dt_float32:
            kernels::trilu(input.buffer().as_span<float>().data(), output.buffer().as_span<float>().data(),
                input.shape(), rnode.upper(), rnode.k())
                .unwrap_or_throw();
            break;
        default:
            throw std::runtime_error("unsupported dtype for topk: " + std::string(datatype_names(datatype)));
        } });

    register_evaluator(op_gru, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<gru &>(node);
        auto input = context.memory_at(rnode.input());
        auto W = context.memory_at(rnode.w());
        auto R = context.memory_at(rnode.r());
        auto B = context.memory_at(rnode.b());
        auto initial_h = context.memory_at(rnode.initial_h());
        auto output = context.memory_at(rnode.output());
        auto output_h = context.memory_at(rnode.output_h());
        kernels::gru(input.buffer().as_span<float>().data(), W.buffer().as_span<float>().data(), R.buffer().as_span<float>().data(),
            B.buffer().as_span<float>().data(), initial_h.buffer().as_span<float>().data(), output.buffer().as_span<float>().data(), output_h.buffer().as_span<float>().data(),
            input.shape(), W.shape(), rnode.direction(), rnode.linear_before_reset())
            .unwrap_or_throw(); });

    register_evaluator(op_tflite_detection_postprocess, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<tflite_detection_postprocess &>(node);
        auto box = context.memory_at(rnode.boxes());
        auto score = context.memory_at(rnode.scores());
        auto anchor = context.memory_at(rnode.anchors());
        auto output_locations = context.memory_at(rnode.output_locations());
        auto output_classes = context.memory_at(rnode.output_classes());
        auto output_scores = context.memory_at(rnode.output_scores());
        auto output_num_detections = context.memory_at(rnode.output_num_detections());
        kernels::tflite_detection_postprocess(box.buffer().as_span<float>().data(), score.buffer().as_span<float>().data(), anchor.buffer().as_span<float>().data(),
            output_locations.buffer().as_span<float>().data(), output_classes.buffer().as_span<float>().data(), output_scores.buffer().as_span<float>().data(), output_num_detections.buffer().as_span<float>().data(),
            box.shape(), score.shape(), anchor.shape(), rnode.max_detections(), rnode.max_classes_per_detection(), 
            rnode.detections_per_class(), rnode.use_regular_non_max_suppression(), rnode.nms_score_threshold(), rnode.nms_iou_threshold(),
            rnode.num_classes(), rnode.y_scale(), rnode.x_scale(), rnode.h_scale(), rnode.w_scale())
            .unwrap_or_throw(); });

    register_evaluator(op_gather_elements, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<gather_elements &>(node);
        auto input = context.memory_at(rnode.input());
        auto indices = context.memory_at(rnode.indices());
        auto output = context.memory_at(rnode.output());
        auto input_datatype = rnode.input().type();

        switch (input_datatype)
        {
        case dt_float32:
            kernels::gather_elements(input.buffer().as_span<float>().data(), indices.buffer().as_span<int64_t>().data(), output.buffer().as_span<float>().data(),
                input.shape(), indices.shape(), rnode.axis())
                .unwrap_or_throw();
            break;
        default:
            throw std::runtime_error("unsupported dtype for gather_elements: " + std::string(datatype_names(input_datatype)));
        }
    });

    register_evaluator(op_instancenorm, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<instancenorm &>(node);

        auto input = context.memory_at(rnode.input());
        auto scale = context.memory_at(rnode.scale());
        auto bias = context.memory_at(rnode.bias());
        auto output = context.memory_at(rnode.output());

        auto output_type = rnode.output().type();
        switch (output_type)
        {
        case dt_float32:
            kernels::instancenorm(input.buffer().as_span<float>().data(), output.buffer().as_span<float>().data(),
                scale.buffer().as_span<float>().data(), bias.buffer().as_span<float>().data(), input.shape(),
                rnode.epsilon())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for layernorm: " + std::string(datatype_names(output_type));
        } });

    register_evaluator(op_layernorm, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<layernorm &>(node);

        auto input = context.memory_at(rnode.input());
        auto scale = context.memory_at(rnode.scale());
        auto bias = context.memory_at(rnode.bias());
        auto output = context.memory_at(rnode.output());

        auto output_type = rnode.output().type();
        switch (output_type)
        {
        case dt_float32:
            kernels::layernorm(input.buffer().as_span<float>().data(), output.buffer().as_span<float>().data(),
             scale.buffer().as_span<float>().data(), bias.buffer().as_span<float>().data(), input.shape(),
                rnode.axis(), rnode.epsilon())
                .unwrap_or_throw();
            break;
        default:
            std::cerr << "unsupported dtype for layernorm: " + std::string(datatype_names(output_type));
        } });

    register_evaluator(op_compress, [](ir::node &node, function_evaluate_context &context) {
        auto &rnode = static_cast<compress &>(node);
        auto input = context.memory_at(rnode.input());
        auto condition = context.memory_at(rnode.condition());
        auto output = context.memory_at(rnode.output());
        kernels::compress(input.buffer().as_span<float>().data(), condition.buffer().as_span<uint8_t>().data(), output.buffer().as_span<float>().data(),
            input.shape(), condition.shape(), rnode.axis())
            .unwrap_or_throw(); });
}

}
