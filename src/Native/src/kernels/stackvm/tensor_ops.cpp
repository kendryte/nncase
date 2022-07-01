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
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t scale,
    [[maybe_unused]] value_t bias, [[maybe_unused]] value_t input_mean,
    [[maybe_unused]] value_t input_var, [[maybe_unused]] value_t epsilon,
    [[maybe_unused]] value_t momentum, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::clamp(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t min,
    [[maybe_unused]] value_t max, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
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

result<value_t> nncase::kernels::stackvm::conv2d_transpose(
    [[maybe_unused]] pad_mode_t pad_mode, [[maybe_unused]] value_t input,
    [[maybe_unused]] value_t weights, [[maybe_unused]] value_t bias,
    [[maybe_unused]] value_t output_shape, [[maybe_unused]] value_t stride,
    [[maybe_unused]] value_t padding, [[maybe_unused]] value_t output_padding,
    [[maybe_unused]] value_t dilation, [[maybe_unused]] value_t groups,
    [[maybe_unused]] value_t fused_clamp, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::expand(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t shape,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::flatten(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t axis,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
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

result<value_t> nncase::kernels::stackvm::hard_sigmoid(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t alpha,
    [[maybe_unused]] value_t beta, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::instance_normalization(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t scale,
    [[maybe_unused]] value_t bias, [[maybe_unused]] value_t epsilon,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::l2_normalization(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::log_softmax(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t axis,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::lp_normalization(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t axis,
    [[maybe_unused]] value_t p, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::lrn(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t alpha,
    [[maybe_unused]] value_t beta, [[maybe_unused]] value_t bias,
    [[maybe_unused]] value_t size, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
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

result<value_t> nncase::kernels::stackvm::normal(
    [[maybe_unused]] typecode_t type, [[maybe_unused]] value_t mean,
    [[maybe_unused]] value_t scale, [[maybe_unused]] value_t seed,
    [[maybe_unused]] value_t shape, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::normal_like(
    [[maybe_unused]] typecode_t type, [[maybe_unused]] value_t input,
    [[maybe_unused]] value_t mean, [[maybe_unused]] value_t scale,
    [[maybe_unused]] value_t seed, [[maybe_unused]] value_t shape,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
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

result<value_t> nncase::kernels::stackvm::selu(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t alpha,
    [[maybe_unused]] value_t gamma, [[maybe_unused]] value_t output,
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
    return err(std::errc::not_supported);
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

result<value_t> nncase::kernels::stackvm::softmax(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t axis,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    try_f32_input(in_mem, input);
    try_f32_output(out_mem, output, input_tensor->dtype(),
                   input_tensor->shape());
    try_positive_axis(axis_value, axis, input_tensor);
    try_(reference::softmax(in_mem, out_mem, input_tensor->shape(),
                            input_tensor->strides(), output_tensor->strides(),
                            axis_value, 1.f));
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::softplus([[maybe_unused]] value_t input,
                                   [[maybe_unused]] value_t output,
                                   [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::softsign([[maybe_unused]] value_t input,
                                   [[maybe_unused]] value_t output,
                                   [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
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

result<value_t> nncase::kernels::stackvm::squeeze(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t dim,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
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

result<value_t> nncase::kernels::stackvm::uniform(
    [[maybe_unused]] typecode_t type, [[maybe_unused]] value_t high,
    [[maybe_unused]] value_t low, [[maybe_unused]] value_t seed,
    [[maybe_unused]] value_t shape, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::uniform_like(
    [[maybe_unused]] typecode_t type, [[maybe_unused]] value_t input,
    [[maybe_unused]] value_t high, [[maybe_unused]] value_t low,
    [[maybe_unused]] value_t seed, [[maybe_unused]] value_t shape,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
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
    auto out_shape = where_infer_shape(cond_tensor->shape(), x_tensor->shape(), y_tensor->shape());
    auto dt = x_tensor->dtype();
    try_output(out_mem, output, dt, out_shape);
    try_(reference::where(
        dt, cond_mem, x_mem, y_mem, out_mem, cond_tensor->shape(),
        x_tensor->shape(), y_tensor->shape(), out_shape, cond_tensor->strides(),
        x_tensor->strides(), y_tensor->strides(), output_tensor->strides()));
    finish;
}
