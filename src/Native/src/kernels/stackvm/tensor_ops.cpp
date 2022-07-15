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
#include "shape_infer.h"
#include <cstring>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/opt_ops.h>
#include <nncase/kernels/stackvm/ref_ops.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<value_t> nncase::kernels::stackvm::batch_normalization(
    value_t input, value_t scale, value_t bias, value_t input_mean,
    value_t input_var, value_t epsilon, [[maybe_unused]] value_t momentum,
    value_t output, [[maybe_unused]] kernel_context &context) {
    try_f32_input(input_mem, input);
    try_f32_input(scale_mem, scale);
    try_f32_input(bias_mem, bias);
    try_f32_input(mean_mem, input_mean);
    try_f32_input(var_mem, input_var);
    try_float_scalar(eps, epsilon);
    try_f32_output(output_mem, output, input_tensor->shape());
    try_(reference::batchnorm(input_mem, scale_mem, bias_mem, mean_mem, var_mem,
                              output_mem, input_tensor->shape(),
                              input_tensor->strides(), output_tensor->strides(),
                              eps));
    finish;
}

result<value_t> kernels::stackvm::binary(binary_op_t binary_op, value_t lhs,
                                         value_t rhs, value_t output,
                                         kernel_context &context) {
    try_input(lhs_mem, lhs);
    try_input(rhs_mem, rhs);
    if (!cmp_dt(lhs_tensor, rhs_tensor)) {
        return err(nncase_errc::datatype_mismatch);
    }

    try_typecode(typecode, lhs_tensor);
    auto out_shape = detail::get_binary_output_shape(lhs_tensor->shape(),
                                                     rhs_tensor->shape());
    try_output(out_mem, output, lhs_tensor->dtype(), out_shape);
    try_(reference::binary(
        typecode, binary_op, lhs_mem, rhs_mem, out_mem, lhs_tensor->shape(),
        lhs_tensor->strides(), rhs_tensor->shape(), rhs_tensor->strides(),
        output_tensor->shape(), output_tensor->strides(), context));
    return ok(output);
}

result<value_t> kernels::stackvm::broadcast(value_t input, value_t shape,
                                            value_t output,
                                            kernel_context &context) {
    try_input(input_mem, input);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    try_dims(out_shape, shape);
    try_output(out_mem, output, dtype, out_shape);
    try_(reference::broadcast(typecode, input_mem, out_mem,
                              input_tensor->shape(), input_tensor->strides(),
                              output_tensor->shape(), output_tensor->strides(),
                              context));
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::clamp(value_t input, value_t min, value_t max,
                                value_t output,
                                [[maybe_unused]] kernel_context &context) {
    try_input(input_mem, input);
    try_input(min_mem, min);
    try_input(max_mem, max);
    try_output_like_input(output_mem, output, input_tensor);
    try_var(typecode, to_typecode(input_tensor->dtype()));
    try_(reference::clamp(typecode, input_mem, min_mem, max_mem, output_mem,
                          input_tensor->shape(), input_tensor->strides(),
                          output_tensor->strides()));
    finish;
}

result<value_t> nncase::kernels::stackvm::concat(value_t input, value_t axis,
                                                 value_t output,
                                                 kernel_context &context) {
    try_tuple_input(inputs_mem, input);
    try_var(shapes, get_shapes(input_tuple));
    try_var(strides, get_strides(input_tuple));
    try_tuple_field0(input0, input_tuple);
    auto dtype = input0->dtype();
    try_positive_axis_with_rank(axis_value, axis, input0->shape().size());
    auto out_shape = concat_infer_shape(shapes, axis_value);
    try_output(out_mem, output, dtype, out_shape);
    auto concat_dims = dims_t();
    if (input0->shape().size() != 0) {
        for (int i = 0; i < input_tuple->fields().size(); ++i) {
            try_var(in, input_tuple->fields()[i].as<tensor>());
            concat_dims.push_back(in->shape()[axis_value]);
        }
    } else {
        concat_dims = dims_t(input_tuple->fields().size(), 1);
    }

    try_(reference::concat(dtype, inputs_mem, out_mem, output_tensor->shape(),
                           strides, output_tensor->strides(), axis_value,
                           concat_dims, context));
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::constant_of_shape(
    value_t shape, value_t value, value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_dims(out_shape, shape);
    try_input(value_mem, value);
    try_output(out_mem, output, value_tensor->dtype(), out_shape);
    try_(reference::constant_of_shape(value_tensor->dtype(), value_mem, out_mem,
                                      out_shape));
    finish;
}

result<value_t> nncase::kernels::stackvm::conv2d(
    pad_mode_t pad_mode, value_t input, value_t weights, value_t bias,
    value_t stride, value_t padding, value_t dilation, value_t groups,
    value_t fused_clamp, value_t output, kernel_context &context) {
    if (pad_mode != pad_mode_t::constant) {
        return err(nncase_errc::runtime_not_found);
    }
    try_f32_input(input_mem, input);
    try_f32_input(weights_mem, weights);
    try_f32_input(bias_mem, bias);
    try_strides(strides_value, stride);
    try_paddings(pads, padding);
    try_to_integer(groups_value, groups);
    try_strides(strides, stride);
    try_strides(dilations, dilation);
    try_f32_input(fused_clamp_value, fused_clamp);
    auto out_shape =
        conv2d_infer_shape(input_tensor->shape(), weights_tensor->shape(),
                           strides_value, dilations, pads);
    try_f32_output(out_mem, output, out_shape);
    try_(reference::conv2d(
        input_mem, weights_mem, bias_mem, out_mem, input_tensor->shape(),
        input_tensor->strides(), weights_tensor->shape(),
        weights_tensor->strides(), bias_tensor->strides(),
        output_tensor->strides(), pads[0], pads[1], groups_value, strides[0],
        strides[1], dilations[0], dilations[1],
        value_range<float>{fused_clamp_value[0], fused_clamp_value[1]},
        context));
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::conv2d_transpose(
    pad_mode_t pad_mode, value_t input, value_t weights, value_t bias,
    value_t output_shape, value_t stride, value_t padding,
    [[maybe_unused]] value_t output_padding, value_t dilation, value_t groups,
    value_t fused_clamp, value_t output, kernel_context &context) {
    if (pad_mode != pad_mode_t::constant) {
        return err(nncase_errc::runtime_not_found);
    }
    try_f32_input(input_mem, input);
    try_f32_input(weights_mem, weights);
    try_f32_input(bias_mem, bias);
    try_strides(strides_value, stride);
    try_paddings(pads, padding);
    try_to_integer(groups_value, groups);
    try_strides(strides, stride);
    try_strides(dilations, dilation);
    try_f32_input(fused_clamp_value, fused_clamp);
    try_dims(out_shape, output_shape);
    try_f32_output(out_mem, output, out_shape);
    try_(reference::conv2d(
        input_mem, weights_mem, bias_mem, out_mem, input_tensor->shape(),
        input_tensor->strides(), weights_tensor->shape(),
        weights_tensor->strides(), bias_tensor->strides(),
        output_tensor->strides(), pads[0], pads[1], groups_value, strides[0],
        strides[1], dilations[0], dilations[1],
        value_range<float>{fused_clamp_value[0], fused_clamp_value[1]},
        context));
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::expand(value_t input, value_t shape, value_t output,
                                 [[maybe_unused]] kernel_context &context) {
    try_input(input_mem, input);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    try_dims(out_shape, shape);
    try_output(out_mem, output, dtype, out_shape);
    try_(reference::broadcast(typecode, input_mem, out_mem,
                              input_tensor->shape(), input_tensor->strides(),
                              output_tensor->shape(), output_tensor->strides(),
                              context));
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::flatten(value_t input, value_t axis, value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_var(in_tensor, input.as<tensor>());
    auto in_shape = in_tensor->shape();
    not_impl_no_contiguous(in_tensor);
    try_positive_axis(axis_value, axis, in_tensor);
    auto new_shape = flatten_infer_shape(in_shape, axis_value);
    output = tensor_reshape(in_tensor, new_shape);
    finish;
}

result<value_t> nncase::kernels::stackvm::gather(value_t input, value_t axis,
                                                 value_t index, value_t output,
                                                 kernel_context &context) {
    try_input(input_mem, input);
    try_input(index_mem, index);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    try_positive_axis(axis_value, axis, input_tensor);
    auto out_shape = gather_infer_shape(input_tensor->shape(),
                                        index_tensor->shape(), axis_value);
    try_output(out_mem, output, dtype, out_shape);

    //    if(is_contiguous(input_tensor->shape(), input_tensor->strides())) {
    //
    //    } else {
    try_(reference::gather(typecode, input_mem, out_mem, input_tensor->shape(),
                           output_tensor->shape(), input_tensor->strides(),
                           output_tensor->strides(), index_tensor->dtype(),
                           index_mem, index_tensor->shape(), axis_value,
                           context));
    //    }
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::gather_nd(value_t input,
                                                    value_t batch_dims,
                                                    value_t index,
                                                    value_t output,
                                                    kernel_context &context) {
    try_input(input_mem, input);
    try_input(index_mem, index);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    try_to_scalar(batch_dims_value, batch_dims, int64_t);
    auto out_shape = gather_nd_infer_shape(
        input_tensor->shape(), index_tensor->shape(), batch_dims_value);
    try_output(out_mem, output, dtype, out_shape);
    CONTIGUOUS_KERNEL(gather_nd, input_tensor, typecode, input_mem, out_mem,
                      input_tensor->shape(), output_tensor->shape(),
                      input_tensor->strides(), output_tensor->strides(),
                      index_tensor->dtype(), index_mem, index_tensor->shape(),
                      batch_dims_value, context);
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::get_item(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t index,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    // todo: not finish
    try_var(tuples, input.as<tuple>());
    try_to_integer(index_value, index);
    auto target = tuples->fields()[index_value];
    output = target;
    finish;
}

result<value_t> nncase::kernels::stackvm::instance_normalization(
    value_t input, value_t scale, value_t bias, value_t epsilon, value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_f32_input(input_mem, input);
    try_f32_input(scale_mem, scale);
    try_f32_input(bias_mem, bias);
    try_float_scalar(eps, epsilon);
    try_f32_output(output_mem, output, input_tensor->shape());
    try_(reference::instance_norm(
        input_mem, scale_mem, bias_mem, output_mem, input_tensor->shape(),
        input_tensor->strides(), output_tensor->strides(), eps));
    finish;
}

result<value_t> nncase::kernels::stackvm::l2_normalization(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::log_softmax(
    value_t input, value_t axis, value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_f32_input(in_mem, input);
    try_f32_output(out_mem, output, input_tensor->shape());
    try_positive_axis(axis_value, axis, input_tensor);
    try_(reference::softmax(in_mem, out_mem, input_tensor->shape(),
                            input_tensor->strides(), output_tensor->strides(),
                            axis_value, 1.f, true));
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::lp_normalization(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t axis,
    [[maybe_unused]] value_t p, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::lrn(value_t input, value_t alpha, value_t beta,
                              value_t bias, value_t size, value_t output,
                              [[maybe_unused]] kernel_context &context) {
    try_f32_in_mem(input);
    try_float_scalar_v(alpha);
    try_float_scalar_v(beta);
    try_float_scalar_v(bias);
    try_to_integer(size_value, size);
    auto out_shape = input_tensor->shape();
    try_f32_out_mem(output, out_shape);
    try_(reference::lrn(input_mem, alpha_value, beta_value, bias_value, size_value,
                        output_mem, input_tensor->shape(),
                        input_tensor->strides(),
                        runtime::get_default_strides(out_shape)));
    finish;
}

result<value_t> nncase::kernels::stackvm::lstm(
    [[maybe_unused]] lstmdirection_t direction,
    [[maybe_unused]] lstmlayout_t layout,
    [[maybe_unused]] std::vector<std::string> activations,
    [[maybe_unused]] value_t x, [[maybe_unused]] value_t w,
    [[maybe_unused]] value_t r, [[maybe_unused]] value_t b,
    [[maybe_unused]] value_t sequence_lens, [[maybe_unused]] value_t initial_h,
    [[maybe_unused]] value_t initial_c, [[maybe_unused]] value_t p,
    [[maybe_unused]] value_t activation_alpha,
    [[maybe_unused]] value_t activation_beta, [[maybe_unused]] value_t clip,
    [[maybe_unused]] value_t hidden_size, [[maybe_unused]] value_t input_forget,
    [[maybe_unused]] value_t output_size, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::normal(typecode_t type, value_t mean, value_t scale,
                                 value_t seed, value_t shape, value_t output,
                                 [[maybe_unused]] kernel_context &context) {
    try_float_scalar(mean_value, mean);
    try_float_scalar(scale_value, scale);
    try_float_scalar(seed_value, seed);
    try_dims(out_shape, shape);
    try_output(out_mem, output, dt_float32, out_shape);
    try_(reference::random_normal(type, out_mem, out_shape, mean_value,
                                  scale_value, seed_value));
    finish;
}

result<value_t> nncase::kernels::stackvm::normal_like(
    typecode_t type, value_t input, value_t mean, value_t scale, value_t seed,
    value_t output, [[maybe_unused]] kernel_context &context) {
    to_tensor(in_tensor, input);
    try_float_scalar(mean_value, mean);
    try_float_scalar(scale_value, scale);
    try_float_scalar(seed_value, seed);
    auto out_shape = in_tensor->shape();
    try_output(out_mem, output, dt_float32, out_shape);
    try_(reference::random_normal(type, out_mem, out_shape, mean_value,
                                  scale_value, seed_value));
    finish;
}

result<value_t>
nncase::kernels::stackvm::prod([[maybe_unused]] value_t input,
                               [[maybe_unused]] value_t output,
                               [[maybe_unused]] kernel_context &context) {

    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::quant_param_of(
    [[maybe_unused]] quant_mode_t quant_mode, [[maybe_unused]] value_t range,
    [[maybe_unused]] value_t bits, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::range(
    [[maybe_unused]] value_t begin, [[maybe_unused]] value_t end,
    [[maybe_unused]] value_t step, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
#define GET_VALUE(_dtype, _in_type)                                            \
    if (cmp_type<_in_type>(_dtype)) {                                          \
        try_input_with_ty(begin_value, begin, _in_type);                       \
        try_input_with_ty(end_value, end, _in_type);                           \
        try_input_with_ty(step_value, step, _in_type);                         \
        auto count =                                                           \
            (dims_t::value_type)((*end_value - *begin_value) / *step_value);   \
        try_output(out_mem, output, _dtype, dims_t{count});                    \
        auto _out_ptr = OUT_CAST(_in_type, out_mem);                           \
        for (int i = 0; i < count; ++i) {                                      \
            auto v = *begin_value + i * *step_value;                           \
            *(_out_ptr + i) = v;                                               \
        }                                                                      \
    }

    try_var(data_tensor, begin.as<tensor>());
    auto dt = data_tensor->dtype();
    GET_VALUE(dt, int32_t);
    GET_VALUE(dt, uint32_t);
    GET_VALUE(dt, int64_t);
    GET_VALUE(dt, uint64_t);
    GET_VALUE(dt, float);
    finish;
}

result<value_t>
nncase::kernels::stackvm::range_of([[maybe_unused]] value_t input,
                                   [[maybe_unused]] value_t output,
                                   [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::reduce(
    reduce_op_t reduce_op, value_t input, value_t axes, value_t init_value,
    value_t keep_dims, value_t output, kernel_context &context) {
    try_input(in_mem, input);
    try_positive_axes(axes_value, axes, input_tensor->shape().size());
    try_to_scalar(keep_dims_value, keep_dims, bool);
    try_input(init_v, init_value);
    try_typecode(typecode, input_tensor);
    auto out_shape =
        reduce_infer_shape(input_tensor->shape(), axes_value, keep_dims_value);
    try_output(out_mem, output, input_tensor->dtype(), out_shape);

    try_(reference::reduce(typecode, reduce_op, init_v, in_mem, out_mem,
                           input_tensor->shape(), axes_value,
                           input_tensor->strides(), output_tensor->strides(),
                           keep_dims_value, context));
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::relu6([[maybe_unused]] value_t input,
                                [[maybe_unused]] value_t output,
                                [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::require(
    [[maybe_unused]] std::string message, [[maybe_unused]] value_t predicate,
    [[maybe_unused]] value_t value, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::reshape(value_t input, value_t shape, value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_var(in_tensor, input.as<tensor>());
    // dim maybe neg
    try_axes(shape_value, shape);
    auto new_shape = reshape_shape_infer(in_tensor->shape(), shape_value);
    not_impl_no_contiguous(in_tensor);
    output = tensor_reshape(in_tensor, new_shape);
    finish;
}

result<value_t> nncase::kernels::stackvm::reverse_sequence(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t seq_lens,
    [[maybe_unused]] value_t batch_axis, [[maybe_unused]] value_t time_axis,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::select(
    [[maybe_unused]] value_t predicate, [[maybe_unused]] value_t true_value,
    [[maybe_unused]] value_t false_value, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::shape_of(value_t input, value_t output,
                                   [[maybe_unused]] kernel_context &context) {
    try_var(in_tensor, input.as<tensor>());
    auto r = in_tensor->shape().size();
    try_output(out_mem, output, dt_int64, dims_t{r});
    auto out = reinterpret_cast<int64_t *>(out_mem);
    for (int i = 0; i < r; ++i) {
        out[i] = in_tensor->shape()[i];
    }
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::size_of([[maybe_unused]] value_t input,
                                  [[maybe_unused]] value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_var(in_tensor, input.as<tensor>());
    try_output(out_mem, output, dt_int64, dims_t{});
    *OUT_CAST(int64_t, out_mem) = compute_size(in_tensor);
    finish;
}

inline std::tuple<axes_t, axes_t, axes_t>
slice_fill(const dims_t &in_shape, axes_t &begins_value, axes_t &ends_value,
           axes_t &strides_value, axes_t axes_value) {
    auto ndim = in_shape.size();
    axes_t begin_values(ndim, 0);
    axes_t end_values(in_shape.begin(), in_shape.end());
    axes_t strides_values(ndim, 1);
    for (auto i = 0; i < ndim; ++i) {
        const auto it = std::find_if(axes_value.begin(), axes_value.end(),
                                     [i, ndim](const auto axis) {
                                         return positive_index(axis, ndim) == i;
                                     });
        if (it != axes_value.end()) {
            auto idx = std::distance(axes_value.begin(), it);
            auto max = static_cast<int>(in_shape[i]);
            auto min = (-1) * max - 1;

            // check starts
            begin_values[i] = begins_value[idx] < min   ? min
                              : begins_value[idx] > max ? max
                                                        : begins_value[idx];

            // check stops
            end_values[i] = ends_value[idx] < min   ? min
                            : ends_value[idx] > max ? max
                                                    : ends_value[idx];

            // check steps
            if (!strides_value.empty()) {
                assert(strides_value[idx] != 0);
                strides_values[i] = strides_value[idx];
            }

            // fixup begin_values
            if ((strides_values[i] > 0 && end_values[i] > begin_values[i]) ||
                (strides_values[i] < 0 && end_values[i] < begin_values[i])) {
                begin_values[i] =
                    begin_values[i] == min ? min + 1 : begin_values[i];
                begin_values[i] =
                    begin_values[i] == max ? max - 1 : begin_values[i];
            }
            if (begin_values[i] < 0)
                begin_values[i] += max;
            if (end_values[i] < 0)
                end_values[i] += max;
        }
    }
    return std::tuple(begin_values, end_values, strides_values);
}

result<value_t> nncase::kernels::stackvm::slice(value_t input, value_t begins,
                                                value_t ends, value_t axes,
                                                value_t strides, value_t output,
                                                kernel_context &context) {
    try_input(in_mem, input);
    try_axes(begins_value, begins);
    try_axes(ends_value, ends);
    try_axes(axes_value, axes);
    try_axes(strides_value, strides);
    auto in_shape = input_tensor->shape();

    auto &&[begin_values, end_values, strides_values] = slice_fill(
        in_shape, begins_value, ends_value, strides_value, axes_value);
    auto out_shape =
        slice_infer_shape(in_shape, begin_values, end_values, strides_values);
    try_output(out_mem, output, input_tensor->dtype(), out_shape);
    try_(reference::slice(input_tensor->dtype(), in_mem, out_mem, in_shape,
                          input_tensor->strides(), output_tensor->strides(),
                          begin_values, end_values, strides_values, context));
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::softmax(value_t input, value_t axis, value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_f32_input(in_mem, input);
    try_f32_output(out_mem, output, input_tensor->shape());
    try_positive_axis(axis_value, axis, input_tensor);
    try_(reference::softmax(in_mem, out_mem, input_tensor->shape(),
                            input_tensor->strides(), output_tensor->strides(),
                            axis_value, 1.f));
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::space_to_batch(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t block_shape,
    [[maybe_unused]] value_t paddings, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::split(value_t input, value_t axis,
                                                value_t sections,
                                                value_t output,
                                                kernel_context &context) {
    try_input(in_mem, input);
    try_positive_axis(axis_value, axis, input_tensor);
    try_dims(sections_value, sections);
    auto shapes =
        split_shape_infer(input_tensor->shape(), axis_value, sections_value);
    try_tuple_output(outputs, output, input_tensor->dtype(), shapes);
    try_var(out_strides, get_strides(output_tuple));
    try_ref(split, input_tensor->dtype(), in_mem, outputs,
            input_tensor->shape(), input_tensor->strides(), out_strides,
            axis_value, sections_value, context);
    finish;
}

result<value_t>
nncase::kernels::stackvm::squeeze(value_t input, value_t dim, value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_var(in_tensor, input.as<tensor>());
    auto in_shape = in_tensor->shape();
    not_impl_no_contiguous(in_tensor);
    try_axes(axes, dim);
    auto new_shape = squeeze_infer_shape(in_shape, axes);
    output = tensor_reshape(in_tensor, new_shape);
    finish;
}

result<value_t> nncase::kernels::stackvm::stack(value_t inputs, value_t axis,
                                                value_t output,
                                                kernel_context &context) {
    try_tuple_input(inputs_value, inputs);
    try_tuple_field0(input0, inputs_tuple);
    try_positive_axis(axis_value, axis, input0);
    auto out_shape = stack_infer_shape(
        input0->shape(), inputs_tuple->fields().size(), axis_value);
    try_output(out_mem, output, input0->dtype(), out_shape);
    try_var(strides, get_strides(inputs_tuple));
    try_(reference::stack(input0->dtype(), inputs_value, out_mem, out_shape,
                          strides, output_tensor->strides(), axis_value,
                          context));
    finish;
}

result<value_t> nncase::kernels::stackvm::tile(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t repeats,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    try_input(in_mem, input);
    try_dims(repeats_value, repeats);
    auto ty = input_tensor->dtype();
    auto out_shape = tile_infer_shape(input_tensor->shape(), repeats_value);
    try_output(out_mem, output, ty, out_shape);
    try_(reference::tile(ty, in_mem, out_mem, input_tensor->shape(), out_shape,
                         input_tensor->strides(), output_tensor->strides(),
                         repeats_value));
    finish;
}

result<value_t>
nncase::kernels::stackvm::uniform(typecode_t type, value_t high, value_t low,
                                  value_t seed, value_t shape, value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_float_scalar(high_value, high);
    try_float_scalar(low_value, low);
    try_float_scalar(seed_value, seed);
    try_dims(out_shape, shape);
    try_output(out_mem, output, dt_float32, out_shape);
    try_(reference::random_uniform(type, out_mem, out_shape, low_value,
                                   high_value, seed_value));
    finish;
}

result<value_t> nncase::kernels::stackvm::uniform_like(
    typecode_t type, value_t input, value_t high, value_t low, value_t seed,
    value_t output, [[maybe_unused]] kernel_context &context) {
    to_tensor(in_tensor, input);
    try_float_scalar(high_value, high);
    try_float_scalar(low_value, low);
    try_float_scalar(seed_value, seed);
    auto out_shape = in_tensor->shape();
    try_output(out_mem, output, dt_float32, out_shape);
    try_(reference::random_uniform(type, out_mem, out_shape, low_value,
                                   high_value, seed_value));
    finish;
}

result<value_t>
nncase::kernels::stackvm::unsqueeze(value_t input, value_t dim, value_t output,
                                    [[maybe_unused]] kernel_context &context) {
    try_var(in_tensor, input.as<tensor>());
    auto in_shape = in_tensor->shape();
    not_impl_no_contiguous(in_tensor);
    try_axes(axes, dim);
    auto new_shape = unsqueeze_infer_shape(in_shape, axes);
    output = tensor_reshape(in_tensor, new_shape);
    finish;
}

result<value_t> nncase::kernels::stackvm::where(
    [[maybe_unused]] value_t cond, [[maybe_unused]] value_t x,
    [[maybe_unused]] value_t y, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_input_with_ty(cond_mem, cond, bool);
    try_input(x_mem, x);
    try_input(y_mem, y);
    auto dt = x_tensor->dtype();
    if (x_tensor->shape()[0] == 0 && y_tensor->shape()[0] == 0 &&
        cmp_type<float>(x_tensor->dtype())) {
        // todo: not finish other rank
        assert(cond_tensor->shape().size() == 1);
        dt = dt_int64;
        auto size = compute_size(cond_tensor->shape());
        std::vector<int64_t> result;
        auto cond_mem_ptr = IN_CAST(bool, cond_mem);
        for (int i = 0; i < size; ++i) {
            if (cond_mem_ptr[i]) {
                result.push_back(i);
            }
        }
        auto out_shape = dims_t{result.size(), cond_tensor->shape().size()};
        try_output(out_mem, output, dt, out_shape);
        memcpy(OUT_CAST(int64_t, out_mem), result.data(),
               result.size() * sizeof(int64_t));
        finish;
    }
    auto out_shape = where_infer_shape(cond_tensor->shape(), x_tensor->shape(),
                                       y_tensor->shape());
    try_output(out_mem, output, dt, out_shape);
    try_(reference::where(
        dt, cond_mem, x_mem, y_mem, out_mem, cond_tensor->shape(),
        x_tensor->shape(), y_tensor->shape(), out_shape, cond_tensor->strides(),
        x_tensor->strides(), y_tensor->strides(), output_tensor->strides()));
    finish;
}

result<value_t> kernels::stackvm::unary(unary_op_t unary_op, value_t input,
                                        value_t output,
                                        kernel_context &context) {
    try_input(input_mem, input);
    try_var(typoecode, to_typecode(input_tensor->dtype()));
    auto dtype = input_tensor->dtype();
    try_output(out_mem, output, dtype, input_tensor->shape());

    try_(reference::unary(typoecode, unary_op, input_mem, out_mem,
                          input_tensor->shape(), input_tensor->strides(),
                          output_tensor->shape(), output_tensor->strides(),
                          context));
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::fake_dequantize(
    [[maybe_unused]] typecode_t target_type, [[maybe_unused]] value_t input,
    [[maybe_unused]] value_t dequant_param, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::fake_quantize(
    [[maybe_unused]] typecode_t target_type, [[maybe_unused]] value_t input,
    [[maybe_unused]] value_t quant_param, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}
