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
#include "optimized/opt_ops.h"
#include "reference/ref_ops.h"
#include "shape_infer.h"
#include <cstring>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/runtime_tensor.h>
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
    try_input(input_mem, input);
    try_input(scale_mem, scale);
    try_input(bias_mem, bias);
    try_input(mean_mem, input_mean);
    try_input(var_mem, input_var);
    try_float_scalar(eps, epsilon);
    try_output_like_input(output_mem, output, input_tensor);
    try_typecode(typecode, input_tensor);
    try_(reference::batchnorm(typecode, input_mem, scale_mem, bias_mem,
                              mean_mem, var_mem, output_mem,
                              input_tensor->shape(), input_tensor->strides(),
                              output_tensor->strides(), eps));
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::layer_norm(
    int32_t axis, float epsilon, [[maybe_unused]] bool use_mean, value_t input,
    value_t scale, value_t bias, value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_input(input_mem, input);
    try_input(scale_mem, scale);
    try_input(bias_mem, bias);
    try_output_like_input(output_mem, output, input_tensor);
    try_typecode(typecode, input_tensor);
    if (typecode == dt_float32) {
        CONTIGUOUS_KERNEL(layer_norm, input_tensor, typecode, input_mem,
                          output_mem, scale_mem, bias_mem,
                          input_tensor->shape(), axis, epsilon);
    } else {
        try_(reference::layer_norm(typecode, input_mem, output_mem, scale_mem,
                                   bias_mem, input_tensor->shape(), axis,
                                   epsilon));
    }
    KERNEL_FINISH;
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
    auto out_shape = kernels::detail::get_binary_output_shape(
        lhs_tensor->shape(), rhs_tensor->shape());
    try_output(out_mem, output, lhs_tensor->dtype(), out_shape);
    CONTIGUOUS_KERNEL(binary, lhs_tensor, typecode, binary_op, lhs_mem, rhs_mem,
                      out_mem, lhs_tensor->shape(), lhs_tensor->strides(),
                      rhs_tensor->shape(), rhs_tensor->strides(),
                      output_tensor->shape(), output_tensor->strides(),
                      context);
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::bitcast(
    [[maybe_unused]] prim_type_t type, [[maybe_unused]] prim_type_t new_type,
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t new_shape,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
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
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::concat(int32_t axis, value_t input,
                                                 value_t output,
                                                 kernel_context &context) {
    try_tuple_input(inputs_mem, input);
    try_var(shapes, get_shapes(input_tuple));
    try_var(strides, get_strides(input_tuple));
    try_tuple_field0(input0, input_tuple);
    auto dtype = input0->dtype();
    auto axis_value = positive_index(axis, input0->shape().size());
    auto out_shape = concat_infer_shape(shapes, axis_value);
    try_output(out_mem, output, dtype, out_shape);
    auto concat_dims = dims_t();
    if (input0->shape().size() != 0) {
        for (size_t i = 0; i < input_tuple->fields().size(); ++i) {
            try_var(in, input_tuple->fields()[i].as<tensor>());
            concat_dims.push_back(in->shape()[axis_value]);
        }
    } else {
        concat_dims = dims_t(input_tuple->fields().size(), 1);
    }
    auto inputs_mem_span =
        gsl::make_span(inputs_mem).as_span<const gsl::byte *const>();

    if (is_contiguous(input0) && axis_value < 4) {
        try_(optimized::concat(
            dtype, inputs_mem_span, out_mem, output_tensor->shape(), strides,
            output_tensor->strides(), axis_value, concat_dims, context))
    } else {
        try_(reference::concat(
            dtype, inputs_mem_span, out_mem, output_tensor->shape(), strides,
            output_tensor->strides(), axis_value, concat_dims, context))
    }
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::condition(
    [[maybe_unused]] bool can_fold_const_call,
    [[maybe_unused]] value_t predicate, [[maybe_unused]] value_t value,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
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
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::conv2d(
    pad_mode_t pad_mode, value_t input, value_t weights, value_t bias,
    value_t stride, value_t padding, value_t dilation, value_t groups,
    value_t fused_clamp, value_t output, kernel_context &context) {
    if (pad_mode != pad_mode_t::constant) {
        return err(nncase_errc::runtime_not_found);
    }
    try_input(input_mem, input);
    try_input(weights_mem, weights);
    try_input(bias_mem, bias);
    try_strides(strides_value, stride);
    try_paddings(pads, padding);
    try_to_integer(groups_value, groups);
    try_strides(strides, stride);
    try_strides(dilations, dilation);
    try_f32_input(fused_clamp_value, fused_clamp);
    try_typecode(typecode, input_tensor);
    auto out_shape =
        conv2d_infer_shape(input_tensor->shape(), weights_tensor->shape(),
                           strides_value, dilations, pads);
    try_output(out_mem, output, typecode, out_shape);

    // CONTIGUOUS_KERNEL(
    //     conv2d, input_tensor, input_mem, weights_mem, bias_mem, out_mem,
    //     input_tensor->shape(), input_tensor->strides(),
    //     weights_tensor->shape(), weights_tensor->strides(),
    //     bias_tensor->strides(), output_tensor->strides(), pads[0],
    //     pads[1], groups_value, strides[0], strides[1], dilations[0],
    //     dilations[1], value_range<float>{fused_clamp_value[0],
    //     fused_clamp_value[1]}, context);
    try_(reference::conv2d(
        typecode, input_mem, weights_mem, bias_mem, out_mem,
        input_tensor->shape(), input_tensor->strides(), weights_tensor->shape(),
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
    value_t fused_clamp, value_t output,
    [[maybe_unused]] kernel_context &context) {
    if (pad_mode != pad_mode_t::constant) {
        return err(nncase_errc::runtime_not_found);
    }
    try_input(input_mem, input);
    try_input(weights_mem, weights);
    try_input(bias_mem, bias);
    try_strides(strides_value, stride);
    try_paddings(pads, padding);
    try_to_integer(groups_value, groups);
    try_strides(strides, stride);
    try_strides(dilations, dilation);
    try_f32_input(fused_clamp_value, fused_clamp);
    try_dims(out_shape, output_shape);
    try_typecode(typecode, input_tensor);
    try_output(out_mem, output, typecode, out_shape);
    try_(reference::conv2d_transpose(
        typecode, input_mem, out_mem, weights_mem, bias_mem,
        input_tensor->shape(), groups_value, output_tensor->shape(),
        weights_tensor->shape()[2], weights_tensor->shape()[3], strides[0],
        strides[1], dilations[0], dilations[1], pads[0], pads[1],
        value_range<float>{fused_clamp_value[0], fused_clamp_value[1]}));
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::expand(value_t input, value_t shape, value_t output,
                                 [[maybe_unused]] kernel_context &context) {
    try_input(input_mem, input);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    try_dims(expand_shape, shape);
    auto out_shape = kernels::detail::get_binary_output_shape(
        input_tensor->shape(), expand_shape);
    try_output(out_mem, output, dtype, out_shape);
    try_(reference::expand(typecode, input_mem, out_mem, input_tensor->shape(),
                           input_tensor->strides(), output_tensor->shape(),
                           output_tensor->strides(), context));
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::dequantize(typecode_t target_type,
                                                     value_t input,
                                                     value_t dequant_param,
                                                     value_t output,
                                                     kernel_context &context) {
    try_input(input_mem, input);
    try_output(out_mem, output, target_type, input_tensor->shape());
    try_input_with_value_type(deq_param, dequant_param, quant_param_t);

    CONTIGUOUS_KERNEL(dequantize, input_tensor, input_tensor->dtype(),
                      output_tensor->dtype(), input_mem, out_mem,
                      input_tensor->shape(), input_tensor->strides(),
                      output_tensor->strides(), deq_param->scale,
                      (float)deq_param->zero_point, context);
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
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::gather(int32_t axis, value_t input,
                                                 value_t index, value_t output,
                                                 kernel_context &context) {
    try_input(input_mem, input);
    try_input(index_mem, index);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    // try_positive_axis(axis_value, axis, input_tensor);
    auto axis_value = positive_index(axis, input_tensor->shape().size());
    auto out_shape = gather_infer_shape(input_tensor->shape(),
                                        index_tensor->shape(), axis_value);
    try_output(out_mem, output, dtype, out_shape);
    CONTIGUOUS_KERNEL(gather, input_tensor, typecode, input_mem, out_mem,
                      input_tensor->shape(), output_tensor->shape(),
                      input_tensor->strides(), output_tensor->strides(),
                      index_tensor->dtype(), index_mem, index_tensor->shape(),
                      axis_value, context);
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::gather_elements(value_t input, value_t axis,
                                          value_t indices, value_t output,
                                          kernel_context &context) {
    try_input(input_mem, input);
    try_input(indices_mem, indices);
    auto dtype = input_tensor->dtype();
    try_positive_axis(axis_value, axis, input_tensor);
    auto out_shape = indices_tensor->shape();
    try_output(out_mem, output, dtype, out_shape);
    try_(optimized::gather_elements(
        dtype, input_mem, out_mem, input_tensor->shape(), out_shape,
        input_tensor->strides(), output_tensor->strides(),
        indices_tensor->dtype(), indices_mem, indices_tensor->shape(),
        axis_value, context));
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

result<value_t> nncase::kernels::stackvm::scatter_nd(value_t input,
                                                     value_t indices,
                                                     value_t updates,
                                                     value_t output,
                                                     kernel_context &context) {
    try_input(input_mem, input);
    try_input(indices_mem, indices);
    try_input(updates_memm, updates);
    auto dtype = input_tensor->dtype();
    auto out_shape = input_tensor->shape();
    try_output(out_mem, output, dtype, out_shape);
    try_(reference::scatter_nd(dtype, input_mem, out_mem, input_tensor->shape(),
                               indices_tensor->dtype(), indices_mem,
                               indices_tensor->shape(), updates_memm,
                               updates_tensor->shape(), context));
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::get_item(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t index,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    // todo: not finish
    if (input.is_a<tuple>()) {
        try_var(tuples, input.as<tuple>());
        try_tuple_field0(input0, tuples);
        try_positive_axis_with_rank(index_value, index, input0->shape().size());
        auto target = tuples->fields()[index_value];
        output = target;
        KERNEL_FINISH;
    } else {
        // used for get dim
        try_input(in_mem, input);
        try_axes(begins_value, index);
        for (int i = 0; i < begins_value.size(); ++i) {
            if (begins_value[i] < 0) {
                begins_value[i] += input_tensor->shape()[i];
            }
        }
        if (input_tensor->shape().size() == 1 && begins_value.size() == 1) {
            auto i = begins_value[0];
            auto out_shape = dims_t{};
            try_output(out_mem, output, input_tensor->dtype(), out_shape);
#define RETURN_RESULT(_in_type)                                                \
    if (cmp_type<_in_type>(input_tensor->dtype())) {                           \
        OUT_CAST(_in_type, out_mem)[0] = IN_CAST(_in_type, in_mem)[i];         \
        return ok(output);                                                     \
    }
            RETURN_RESULT_SELECT(RETURN_RESULT);
#undef RETURN_RESULT
            return err(std::errc::not_supported);
        }

        if (input_tensor->shape().size() == 2 && begins_value.size() == 1) {
            auto get_item_index = begins_value[0];
            auto out_shape = dims_t{input_tensor->shape()[1]};
            try_output(out_mem, output, input_tensor->dtype(), out_shape);
            auto size = input_tensor->shape()[1];
#define RETURN_RESULT(_in_type)                                                \
    if (cmp_type<_in_type>(input_tensor->dtype())) {                           \
        for (int i = 0; i < size; ++i) {                                       \
            OUT_CAST(_in_type, out_mem)                                        \
            [i] = IN_CAST(_in_type, in_mem)[get_item_index * size + i];        \
        }                                                                      \
        return ok(output);                                                     \
    }
            RETURN_RESULT_SELECT(RETURN_RESULT);
#undef RETURN_RESULT
            return err(std::errc::not_supported);
        }

        auto n = begins_value.size();
        auto in_shape = input_tensor->shape();
        auto ends_value = axes_t(n, 0);
        auto axes_value = axes_t(n, 0);
        for (size_t i = 0; i < n; ++i) {
            ends_value[i] = begins_value[i] + 1;
            axes_value[i] = i;
        }
        auto strides_value = axes_t(n, 1);

        auto &&[begin_values, end_values, strides_values] = slice_fill(
            in_shape, begins_value, ends_value, strides_value, axes_value);
        auto out_shape = slice_infer_shape(in_shape, begin_values, end_values,
                                           strides_values);
        try_output(out_mem, output, input_tensor->dtype(), out_shape);
        CONTIGUOUS_KERNEL(slice, input_tensor, input_tensor->dtype(), in_mem,
                          out_mem, in_shape, input_tensor->strides(),
                          output_tensor->strides(), begin_values, end_values,
                          strides_values, context);
        output = tensor_reshape(output_tensor,
                                dims_t(out_shape.begin() + n, out_shape.end()));
        KERNEL_FINISH;
    }
}

result<value_t> nncase::kernels::stackvm::instance_normalization(
    value_t input, value_t scale, value_t bias, value_t epsilon, value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_input(input_mem, input);
    try_input(scale_mem, scale);
    try_input(bias_mem, bias);
    try_float_scalar(eps, epsilon);
    try_output_like_input(output_mem, output, input_tensor);
    try_typecode(type, input_tensor);
    try_(reference::instance_norm(
        type, input_mem, scale_mem, bias_mem, output_mem, input_tensor->shape(),
        input_tensor->strides(), output_tensor->strides(), eps));
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::l2_normalization(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::log_softmax(
    value_t input, value_t axis, value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_input(in_mem, input);
    try_output_like_input(out_mem, output, input_tensor);
    try_positive_axis(axis_value, axis, input_tensor);
    try_typecode(type, input_tensor);

    if (type == dt_float32) {
        CONTIGUOUS_KERNEL(log_softmax, input_tensor, type, in_mem, out_mem,
                          input_tensor->shape(), input_tensor->strides(),
                          output_tensor->strides(), axis_value);
    } else {
        try_(reference::log_softmax(
            type, in_mem, out_mem, input_tensor->shape(),
            input_tensor->strides(), output_tensor->strides(), axis_value));
    }

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
    try_in_mem(input);
    try_float_scalar_v(alpha);
    try_float_scalar_v(beta);
    try_float_scalar_v(bias);
    try_to_integer(size_value, size);
    auto out_shape = input_tensor->shape();
    try_typecode(typecode, input_tensor);
    try_out_mem(output, typecode, out_shape);
    try_(reference::lrn(typecode, input_mem, alpha_value, beta_value,
                        bias_value, size_value, output_mem,
                        input_tensor->shape(), input_tensor->strides(),
                        runtime::get_default_strides(out_shape)));
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::lstm(
    lstmdirection_t direction, [[maybe_unused]] lstmlayout_t layout,
    [[maybe_unused]] std::vector<std::string> activations, value_t x, value_t w,
    value_t r, value_t b, value_t sequence_lens, value_t initial_h,
    value_t initial_c, [[maybe_unused]] value_t p,
    [[maybe_unused]] value_t activation_alpha,
    [[maybe_unused]] value_t activation_beta, [[maybe_unused]] value_t clip,
    value_t hidden_size, [[maybe_unused]] value_t input_forget,
    value_t output_size, value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_in_mem(x);
    try_in_mem(w);
    try_in_mem(r);
    try_in_mem(b);
    try_dims_v(sequence_lens);
    try_in_mem(initial_h);
    try_in_mem(initial_c);
    // todo:p
    //    try_f32_in_mem(p);
    try_integer_v(hidden_size);
    try_integer_v(output_size);
    try_typecode(type, x_tensor);
    auto output_shapes = lstm_infer_shape(
        x_tensor->shape(), initial_h_tensor->shape(), initial_c_tensor->shape(),
        direction, layout, hidden_size_value, output_size_value);
    try_tuple_output(out_tuple, output, dt_float32, output_shapes);
    try_(reference::lstm(
        type, x_mem, w_mem, r_mem, b_mem, initial_h_mem, initial_c_mem,
        out_tuple[0], out_tuple[1], out_tuple[2], x_tensor->shape(),
        initial_h_tensor->shape(), initial_c_tensor->shape(), output_shapes[0],
        w_tensor->shape(), r_tensor->shape(), direction));
    KERNEL_FINISH;
}

result<value_t>
nncase::kernels::stackvm::mat_mul(value_t lhs, value_t rhs, value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_input(lhs_mem, lhs);
    try_input(rhs_mem, rhs);
    try_var(out_shape,
            matmul_infer_shape(lhs_tensor->shape(), rhs_tensor->shape()));
    try_output(out_mem, output, lhs_tensor->dtype(), out_shape);
    try_typecode(typecode, lhs_tensor);
    try_(reference::matmul(typecode, lhs_mem, rhs_mem, out_mem,
                           lhs_tensor->shape(), rhs_tensor->shape()));
    return ok(output);
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
    KERNEL_FINISH;
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
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::one_hot(one_hot_mode_t one_hot_mode,
                                                  value_t indices,
                                                  value_t depth, value_t values,
                                                  value_t axis, value_t output,
                                                  kernel_context &context) {
    try_input(onehot_values, values);
    try_var(typecode, to_typecode(values_tensor->dtype()));
    try_to_integer(depth_value, depth);
    try_input(indices_mem, indices);
    try_positive_axis(axis_value, axis, indices_tensor);
    auto out_shape =
        onehot_infer_shape(indices_tensor->shape(), depth_value, axis_value);
    try_output(out_mem, output, typecode, out_shape);

    CONTIGUOUS_KERNEL(one_hot, indices_tensor, typecode,
                      indices_tensor->dtype(), indices_mem, out_mem,
                      indices_tensor->shape(), output_tensor->shape(),
                      output_tensor->strides(), depth_value, onehot_values,
                      axis_value, one_hot_mode, context);
    return ok(output);
}

inline bool is_nop_pad([[maybe_unused]] const paddings_t &paddings) {
#ifdef SKIP_NOP
    return false;
#else
    return std::all_of(paddings.begin(), paddings.end(),
                       [](const padding &p) { return p.sum() == 0; });
#endif
}

result<value_t>
nncase::kernels::stackvm::pad(runtime::stackvm::pad_mode_t pad_mode,
                              value_t input, value_t pads, value_t value,
                              value_t output, kernel_context &context) {
    try_input(input_mem, input);
    try_paddings(paddings, pads);
    if (is_nop_pad(paddings)) {
        return ok(input);
    }
    if (pad_mode == runtime::stackvm::pad_mode_t::reflect) {
        // 在tensorflow和onnxruntime中，pad在reflect mode下
        // 当before_pad/after_pad > in_dim - 1时
        // 均会出现未定义的行为，因此需要约束输入pad大小
        // 针对in_dim-1 < after_pad  <= in_dim-1+before_pad的情况
        // 此处实现结果与onnxruntime对齐
        for (size_t i = 0; i < paddings.size(); ++i) {
            if (paddings[i].before > (input_tensor->shape()[i] - 1) ||
                (paddings[i].after >
                 (input_tensor->shape()[i] + paddings[i].before - 1)))
                return err(std::errc::invalid_argument);
        }
    }
    auto out_shape = pad_infer_shape(input_tensor->shape(), paddings);
    try_output(out_mem, output, input_tensor->dtype(), out_shape);

    try_input(pad_value, value);
    try_(reference::pad(input_tensor->dtype(), input_mem, out_mem,
                        input_tensor->shape(), input_tensor->strides(),
                        output_tensor->strides(), paddings, pad_mode, pad_value,
                        context));

    return ok(output);
}

result<value_t> kernels::stackvm::prelu(value_t input, value_t slope,
                                        value_t output,
                                        kernel_context &context) {
    try_in_mem(input);
    try_in_mem(slope);
    try_output_like_input(out_mem, output, input_tensor);
    try_typecode(type, input_tensor);
    try_(reference::prelu(
        type, input_mem, slope_mem, out_mem, input_tensor->shape(),
        input_tensor->strides(), slope_tensor->shape(), slope_tensor->strides(),
        output_tensor->shape(), output_tensor->strides(), context));
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::prod([[maybe_unused]] value_t input,
                               [[maybe_unused]] value_t output,
                               [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::quantize(typecode_t target_type,
                                                   value_t input,
                                                   value_t quant_param,
                                                   value_t output,
                                                   kernel_context &context) {
    try_input(input_mem, input);
    try_output(out_mem, output, target_type, input_tensor->shape());
    try_input_with_value_type(qp, quant_param, quant_param_t);

    CONTIGUOUS_KERNEL(
        quantize, input_tensor, input_tensor->dtype(), output_tensor->dtype(),
        input_mem, out_mem, input_tensor->shape(), input_tensor->strides(),
        output_tensor->strides(), qp->scale, (float)qp->zero_point, context);
    return ok(output);
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
        for (size_t i = 0; i < count; ++i) {                                   \
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
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::range_of(
    [[maybe_unused]] bool is_range_of_weight, [[maybe_unused]] value_t input,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
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

    CONTIGUOUS_KERNEL(reduce, input_tensor, typecode, reduce_op, init_v, in_mem,
                      out_mem, input_tensor->shape(), axes_value,
                      input_tensor->strides(), output_tensor->strides(),
                      keep_dims_value, context);

    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::relu6([[maybe_unused]] value_t input,
                                [[maybe_unused]] value_t output,
                                [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::require(
    [[maybe_unused]] std::string message,
    [[maybe_unused]] bool can_fold_const_call,
    [[maybe_unused]] value_t predicate, [[maybe_unused]] value_t value,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    try_to_scalar(cond, predicate, bool);
    if (!cond) {
        printf("%s\n", message.data());
        return err(std::errc::invalid_argument);
    }
    output = value;
    KERNEL_FINISH;
}

inline bool is_nop_pad([[maybe_unused]] const std::vector<int> &paddings) {
#ifdef SKIP_NOP
    return false;
#else
    return std::all_of(paddings.begin(), paddings.end(),
                       [](auto &p) { return p == 0; });
#endif
}

result<value_t> nncase::kernels::stackvm::bucket_pad(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t shape,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &) {
    try_dims_v(shape);
    auto in_tensor = input.as<tensor>().expect("input is not a tensor");
    auto in_shape = in_tensor->shape();
    if (compute_size(in_shape) > compute_size(shape_value)) {
        return err(std::errc::invalid_argument);
    }

    auto paddings = std::vector<int>(8);
    auto rank = shape_value.size();
    for (int i = 0; i < rank; ++i) {
        paddings[2 * i + 0] = 0;
        paddings[2 * i + 1] = shape_value[i] - in_shape[i];
    }
    if (is_nop_pad(paddings)) {
        return ok(input);
    }
    auto pads_shape = dims_t{rank, 2};
    auto span = gsl::span(reinterpret_cast<gsl::byte *>(paddings.data()),
                          compute_size(pads_shape) * sizeof(int));
    try_var(pads, hrt::create(dt_int32, pads_shape, span, false,
                              host_runtime_tensor::pool_cpu_only));
#define RUN_PAD                                                                \
    auto data = gsl::span(reinterpret_cast<gsl::byte *>(&pad_value),           \
                          in_tensor->dtype()->size_bytes());                   \
    try_var(pad_v, hrt::create(in_tensor->dtype()->typecode(), dims_t{}, data, \
                               false, host_runtime_tensor::pool_cpu_only));    \
    return nncase::kernels::stackvm::pad(pad_mode_t::constant, input,          \
                                         pads.impl(), pad_v.impl(), output);

    if (runtime::get_bytes(in_tensor->dtype()) == 1) {
        auto pad_value = (int8_t)0;
        RUN_PAD;
    } else if (runtime::get_bytes(in_tensor->dtype()) == 2) {
        auto pad_value = (int16_t)0;
        RUN_PAD;
    }
    if (runtime::get_bytes(in_tensor->dtype()) == 4) {
        auto pad_value = (int32_t)0;
        RUN_PAD;
    }
    if (runtime::get_bytes(in_tensor->dtype()) == 8) {
        auto pad_value = (int64_t)0;
        RUN_PAD;
    }
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::rank([[maybe_unused]] value_t input,
                               [[maybe_unused]] value_t output,
                               [[maybe_unused]] kernel_context &) {
    try_output(out_mem, output, dt_int64, dims_t{});
    OUT_CAST(int64_t, out_mem)[0] = input.as<tensor>().unwrap()->shape().size();
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::index_of(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t value,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &) {
    return err(std::errc::not_supported);
    auto t = input.as<tensor>().unwrap();
    try_output(out_mem, output, dt_int32, dims_t{});
    try_input(input_mem, input);
    try_input(value_mem, value);

#define TRANSLATE_TYPE(_ty)                                                    \
    for (int i = 0; i < compute_size(t->shape()); ++i) {                       \
        if (input_mem[i] == value_mem[0]) {                                    \
            OUT_CAST(int64_t, out_mem)[0] = i;                                 \
            return ok(output);                                                 \
        }                                                                      \
    }                                                                          \
    OUT_CAST(int64_t, out_mem)[0] = -1;                                        \
    return ok(output);

    RETURN_RESULT_SELECT(TRANSLATE_TYPE);
}

result<value_t> nncase::kernels::stackvm::fix_shape(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t shape,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &) {
    return ok(input);
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
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::resize_image(
    image_resize_mode_t resize_mode,
    image_resize_transformation_mode_t transformation_mode,
    [[maybe_unused]] image_resize_nearest_mode_t nearest_mode,
    [[maybe_unused]] bool is_tfresize, value_t input,
    [[maybe_unused]] value_t roi, value_t new_size,
    [[maybe_unused]] value_t cubic_coeff_a,
    [[maybe_unused]] value_t exclude_outside,
    [[maybe_unused]] value_t extrapolation_value, value_t output,
    kernel_context &context) {
    try_input(in_mem, input);
    auto ty = input_tensor->dtype();
    try_var(tycode, to_typecode(ty));
    try_dims(new_size_value, new_size);
    try_output(out_mem, output, input_tensor->dtype(), new_size_value);

    bool align_corner = false;
    bool half_pixel = false;
    if (transformation_mode == image_resize_transformation_mode_t::half_pixel ||
        transformation_mode ==
            image_resize_transformation_mode_t::pytorch_half_pixel) {
        half_pixel = true;
    } else if (transformation_mode ==
               image_resize_transformation_mode_t::align_corners) {
        align_corner = true;
    }
    if (resize_mode == image_resize_mode_t::bilinear) {
        CONTIGUOUS_KERNEL(resize_bilinear, input_tensor, tycode, in_mem,
                          out_mem, input_tensor->shape(),
                          input_tensor->strides(), output_tensor->strides(),
                          new_size_value[2], new_size_value[3], align_corner,
                          half_pixel, context);
    } else if (resize_mode == image_resize_mode_t::nearest_neighbor) {
        auto get_coordinate_func =
            get_coordinate_from_resized(transformation_mode);
        auto get_nearset_func = get_nearest_pixel_from_origin(nearest_mode);
        CONTIGUOUS_KERNEL(resize_nearest_neighbor, input_tensor, tycode, in_mem,
                          out_mem, input_tensor->shape(),
                          input_tensor->strides(), output_tensor->strides(),
                          new_size_value[2], new_size_value[3], align_corner,
                          half_pixel, get_coordinate_func, get_nearset_func,
                          context);
    } else {
        return err(nncase_errc::runtime_not_found);
    }
    // todo: some param not be used
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::reverse_sequence(
    value_t input, value_t seq_lens, value_t batch_axis, value_t time_axis,
    value_t output, [[maybe_unused]] kernel_context &context) {
    try_in_mem(input);
    try_integer_v(batch_axis);
    try_integer_v(time_axis);
    try_dims(seq_lens_value, seq_lens);
    auto out_shape = input_tensor->shape();
    try_out_mem(output, input_tensor->dtype(), out_shape);
    try_(reference::reverse_sequence(
        input_tensor->dtype(), input_mem, output_mem, input_tensor->shape(),
        seq_lens_value, batch_axis_value, time_axis_value,
        input_tensor->strides(), output_tensor->strides()));
    KERNEL_FINISH;
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
    for (size_t i = 0; i < r; ++i) {
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
    KERNEL_FINISH;
}

inline bool is_nop_slice([[maybe_unused]] const axes_t &begin,
                         [[maybe_unused]] const axes_t &end,
                         [[maybe_unused]] const axes_t &axes,
                         [[maybe_unused]] const axes_t &strides,
                         [[maybe_unused]] const dims_t &in_shape) {
#ifdef SKIP_NOP
    return false;
#else
    if (begin.size() != in_shape.size()) {
        return false;
    }
    if (!std::all_of(begin.begin(), begin.end(),
                     [](auto x) { return x == 0; })) {
        return false;
    }
    for (int i = 0; i < in_shape.size(); ++i) {
        if (end[i] < in_shape[i]) {
            return false;
        }
    }
    if (!std::all_of(strides.begin(), strides.end(),
                     [](auto x) { return x == 1; })) {
        return false;
    }
    // todo: check axes
    return true;
#endif
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
    if (is_nop_slice(begins_value, ends_value, axes_value, strides_value,
                     in_shape)) {
        return ok(input);
    }

    auto &&[begin_values, end_values, strides_values] = slice_fill(
        in_shape, begins_value, ends_value, strides_value, axes_value);
    auto out_shape =
        slice_infer_shape(in_shape, begin_values, end_values, strides_values);
    try_output(out_mem, output, input_tensor->dtype(), out_shape);

    bool neg_strides = false;
    for (auto &&stride : strides_value) {
        if (stride < 0) {
            neg_strides = true;
            break;
        }
    }
    if (neg_strides) {
        try_(reference::slice(input_tensor->dtype(), in_mem, out_mem, in_shape,
                              input_tensor->strides(), output_tensor->strides(),
                              begin_values, end_values, strides_values,
                              context));
    } else {
        try_(optimized::slice(input_tensor->dtype(), in_mem, out_mem, in_shape,
                              input_tensor->strides(), output_tensor->strides(),
                              begin_values, end_values, strides_values,
                              context));
    }
    return ok(output);
}

result<value_t>
nncase::kernels::stackvm::softmax(value_t input, value_t axis, value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_input(in_mem, input);
    try_output_like_input(out_mem, output, input_tensor);
    try_positive_axis(axis_value, axis, input_tensor);
    try_typecode(type, input_tensor);
    if (type == dt_float32 || type == dt_float16) {
        CONTIGUOUS_KERNEL(softmax, input_tensor, type, in_mem, out_mem,
                          input_tensor->shape(), input_tensor->strides(),
                          output_tensor->strides(), axis_value, 1.f);
    } else {
        try_(reference::softmax(type, in_mem, out_mem, input_tensor->shape(),
                                input_tensor->strides(),
                                output_tensor->strides(), axis_value, 1.f));
    }
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::space_to_batch(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t block_shape,
    [[maybe_unused]] value_t paddings, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_in_mem(input);
    try_paddings(paddings_value, paddings);
    try_dims_v(block_shape);
    auto out_shape = space_to_batch_shape_infer(
        input_tensor->shape(), block_shape_value, paddings_value);
    try_out_mem(output, input_tensor->dtype(), out_shape);

    try_(reference::space_to_batch(input_tensor->dtype(), input_mem, output_mem,
                                   input_tensor->shape(), block_shape_value,
                                   paddings_value, input_tensor->strides(),
                                   out_shape, output_tensor->strides()));
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::split(value_t input, value_t axis,
                                                value_t sections,
                                                value_t output,
                                                kernel_context &context) {
    try_input(in_mem, input);
    try_positive_axis(axis_value, axis, input_tensor);
    try_dims(sections_value, sections);
    auto in_shape = input_tensor->shape();
    auto in_strides = input_tensor->strides();
    auto shapes = split_shape_infer(in_shape, axis_value, sections_value);
    auto in_place = std::all_of(in_shape.subspan(0, axis_value).begin(),
                                in_shape.subspan(0, axis_value).end(),
                                [](auto x) { return x == 1; });
    if (is_contiguous(input_tensor) && in_place) {
        auto size = shapes.size();
        std::vector<value_t> fields(size);
        auto start = 0;
        for (size_t i = 0; i < size; ++i) {
            auto new_shape = shapes[i];
            auto index = dims_t(new_shape.size(), 0);
            index[axis_value] = start;
            fields[i] = tensor(
                std::in_place, input_tensor->dtype(), new_shape,
                get_default_strides(new_shape),
                buffer_slice(input_tensor->buffer().buffer(),
                             offset(in_strides, index) *
                                 get_bytes(input_tensor->dtype()),
                             get_bytes(input_tensor->dtype(), new_shape)));
            start += sections_value[i];
        }

        output = tuple(std::in_place, std::move(fields));
    } else {
        try_tuple_output(outputs, output, input_tensor->dtype(), shapes);
        try_var(out_strides, get_strides(output_tuple));
        if (is_contiguous(input_tensor)) {
            auto n = in_shape.size();
            auto begins = axes_t(n, 0);
            auto ends = axes_t(in_shape.begin(), in_shape.end());
            auto strides = axes_t(n, 1);
            auto section_index = 0;
            for (int i = 0; i < outputs.size(); ++i) {
                auto out = outputs[i];
                begins[axis_value] = section_index;
                ends[axis_value] = section_index + sections_value[i];
                section_index += sections_value[i];
                try_(optimized::slice(input_tensor->dtype(), in_mem, out,
                                      in_shape, in_strides, out_strides[i],
                                      begins, ends, strides, context));
            }
        } else {
            try_ref(split, input_tensor->dtype(), in_mem, outputs, in_shape,
                    in_strides, out_strides, axis_value, sections_value,
                    context);
        }
    }
    KERNEL_FINISH;
}

result<value_t>
nncase::kernels::stackvm::squeeze(value_t input, value_t dim, value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_var(in_tensor, input.as<tensor>());
    auto in_shape = in_tensor->shape();
    not_impl_no_contiguous(in_tensor);
    try_positive_axes(axes, dim, in_tensor->shape().size());
    auto new_shape = squeeze_infer_shape(in_shape, axes);
    output = tensor_reshape(in_tensor, new_shape);
    KERNEL_FINISH;
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
    try_var(shapes, get_shapes(inputs_tuple));
    auto strides = std::vector<dims_t>(shapes.size());
    for (int i = 0; i < shapes.size(); ++i) {
        strides[i] = get_default_strides(shapes[i]);
    }
    auto inputs_value_span =
        gsl::make_span(inputs_value).as_span<const gsl::byte *const>();
    try_(reference::stack(input0->dtype(), inputs_value_span, out_mem,
                          out_shape, strides, output_tensor->strides(),
                          axis_value, context));
    KERNEL_FINISH;
}

result<value_t>
nncase::kernels::stackvm::tile(value_t input, value_t repeats, value_t output,
                               [[maybe_unused]] kernel_context &context) {
    try_input(in_mem, input);
    try_dims(repeats_value, repeats);
    auto ty = input_tensor->dtype();
    auto out_shape = tile_infer_shape(input_tensor->shape(), repeats_value);
    try_output(out_mem, output, ty, out_shape);
    try_(optimized::tile(ty, in_mem, out_mem, input_tensor->shape(), out_shape,
                         input_tensor->strides(), output_tensor->strides(),
                         repeats_value));
    KERNEL_FINISH;
}

result<value_t>
nncase::kernels::stackvm::top_k(value_t x, value_t k, value_t axis,
                                value_t largest, value_t sorted, value_t output,
                                [[maybe_unused]] kernel_context &context) {
    try_in_mem(x);
    try_integer_v(k);
    try_positive_axis(axis_value, axis, x_tensor);
    auto out_shape = topk_infer_shape(x_tensor->shape(), k_value, axis_value);

    try_(alloc_tuple_output(output, {x_tensor->dtype(), datatype_t::int64},
                            {out_shape, out_shape}));
    try_var(output_tuple, output.as<tuple>());
    try_var(outputs, get_output_data(output_tuple));
    try_var(out_values, output_tuple->fields()[0].as<tensor>());
    try_var(out_indices, output_tuple->fields()[1].as<tensor>());

    try_var(tycode, to_typecode(x_tensor->dtype()));
    try_integer_v(largest);
    try_integer_v(sorted);
    try_(optimized::topk(
        tycode, x_mem, outputs[0], OUT_CAST(int64_t, outputs[1]),
        x_tensor->shape(), x_tensor->strides(), out_values->shape(),
        out_values->strides(), out_indices->shape(), out_indices->strides(),
        k_value, axis_value, largest_value, sorted_value));
    KERNEL_FINISH;
}

result<value_t>
nncase::kernels::stackvm::transpose(value_t input, value_t perm, value_t output,
                                    [[maybe_unused]] kernel_context &context) {
    try_input(input_mem, input);
    auto dt = input_tensor->dtype();
    try_dims(perm_value, perm);
    auto out_shape = transpose_infer_shape(input_tensor->shape(), perm_value);
    try_output(out_mem, output, dt, out_shape);

    if (out_shape.size() == 4) {
        try_(optimized::transpose(dt, input_mem, out_mem, input_tensor->shape(),
                                  perm_value, input_tensor->strides(),
                                  output_tensor->strides(), context));
    } else {
        try_(reference::transpose(dt, input_mem, out_mem, input_tensor->shape(),
                                  perm_value, input_tensor->strides(),
                                  output_tensor->strides()));
    }
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::trilu(value_t input, value_t k,
                                                value_t upper, value_t output,
                                                kernel_context &) {
    try_input(in_mem, input);
    try_integer_v(k);
    try_integer_v(upper);
    try_output(out_mem, output, input_tensor->dtype(), input_tensor->shape());
    try_(reference::trilu(input_tensor->dtype(), in_mem, out_mem,
                          input_tensor->shape(), input_tensor->strides(),
                          output_tensor->strides(), k_value, upper_value == 1))
        KERNEL_FINISH;
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
    KERNEL_FINISH;
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
    KERNEL_FINISH;
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
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::where(
    [[maybe_unused]] bool is_tf_where, [[maybe_unused]] value_t cond,
    [[maybe_unused]] value_t x, [[maybe_unused]] value_t y,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    try_input_with_ty(cond_mem, cond, bool);
    try_input(x_mem, x);
    try_input(y_mem, y);
    auto dt = x_tensor->dtype();
    if ((x_tensor->shape().size() == 0 || x_tensor->shape()[0] == 0) &&
        (y_tensor->shape().size() == 0 || y_tensor->shape()[0]) == 0 &&
        cmp_type<float>(x_tensor->dtype())) {
        // todo: not finish other rank
        assert(cond_tensor->shape().size() == 1);
        dt = dt_int64;
        auto size = compute_size(cond_tensor->shape());
        std::vector<int64_t> result;
        auto cond_mem_ptr = IN_CAST(bool, cond_mem);
        for (size_t i = 0; i < size; ++i) {
            if (cond_mem_ptr[i]) {
                result.push_back(i);
            }
        }
        auto out_shape = dims_t{result.size(), cond_tensor->shape().size()};
        try_output(out_mem, output, dt, out_shape);
        memcpy(OUT_CAST(int64_t, out_mem), result.data(),
               result.size() * sizeof(int64_t));
        KERNEL_FINISH;
    }
    auto out_shape = where_infer_shape(cond_tensor->shape(), x_tensor->shape(),
                                       y_tensor->shape());
    try_output(out_mem, output, dt, out_shape);
    CONTIGUOUS_KERNEL(where, cond_tensor, dt, cond_mem, x_mem, y_mem, out_mem,
                      cond_tensor->shape(), x_tensor->shape(),
                      y_tensor->shape(), out_shape, cond_tensor->strides(),
                      x_tensor->strides(), y_tensor->strides(),
                      output_tensor->strides());
    KERNEL_FINISH;
}

result<value_t> kernels::stackvm::unary(unary_op_t unary_op, value_t input,
                                        value_t output,
                                        kernel_context &context) {
    try_input(input_mem, input);
    try_var(typoecode, to_typecode(input_tensor->dtype()));
    auto dtype = input_tensor->dtype();
    try_output(out_mem, output, dtype, input_tensor->shape());

    if (typoecode != dt_float32) {
        try_(reference::unary(typoecode, unary_op, input_mem, out_mem,
                              input_tensor->shape(), input_tensor->strides(),
                              output_tensor->shape(), output_tensor->strides(),
                              context));
        return ok(output);
    } else {
        CONTIGUOUS_KERNEL(unary, input_tensor, typoecode, unary_op, input_mem,
                          out_mem, input_tensor->shape(),
                          input_tensor->strides(), output_tensor->shape(),
                          output_tensor->strides(), context);
    }
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

// result<value_t> nncase::kernels::stackvm::uninitialized(
//    NNCASE_UNUSED typecode_t dtype,
//    NNCASE_UNUSED runtime::stackvm::memory_location_t memory_location,
//    NNCASE_UNUSED value_t shape, NNCASE_UNUSED value_t output,
//    NNCASE_UNUSED kernel_context &context) {
//    return err(std::errc::not_supported);
//}