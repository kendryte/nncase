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
#pragma once

#define FLOAT_UNARY_IMPL_TEMPLATE(_name, _compute)                             \
    template <class T>                                                         \
    result<void> _name##_impl(                                                 \
        const T *input, T *output, const dims_t &in_shape,                     \
        const strides_t &input_strides, const dims_t &out_shape,               \
        const strides_t &out_strides,                                          \
        NNCASE_UNUSED kernel_context &context) noexcept {                      \
        return apply(out_shape, [&](const dims_t &index) -> result<void> {     \
            const auto in_index =                                              \
                kernels::detail::get_reduced_offset(index, in_shape);          \
            auto src_idx = offset(input_strides, in_index);                    \
            auto dst_idx = offset(out_strides, in_index);                      \
            auto x = input[src_idx];                                           \
            output[dst_idx] = _compute;                                        \
            return ok();                                                       \
        });                                                                    \
    }

#define FLOAT_UNARY_OP_TEMPLATE(_name)                                         \
    result<value_t> nncase::kernels::stackvm::_name(                           \
        value_t input, value_t output, kernel_context &context) {              \
        try_f32_input(input_mem, input);                                       \
        auto dtype = input_tensor->dtype();                                    \
        try_f32_output(out_mem, output, dtype, input_tensor->shape());         \
        try_(_name##_impl(input_mem, out_mem, input_tensor->shape(),           \
                          input_tensor->strides(), output_tensor->shape(),     \
                          output_tensor->strides(), context));                 \
        return ok(output);                                                     \
    }

#define FLOAT_UNARY_TEMPLATE(_name, _compute)                                  \
    FLOAT_UNARY_IMPL_TEMPLATE(_name, _compute)                                 \
    FLOAT_UNARY_OP_TEMPLATE(_name)

#define FLOAT_UNARY_WITH_MUL_IMPL_TEMPLATE(_name, _alpha_name, _compute)       \
    template <class T>                                                         \
    result<void> _name##_impl(                                                 \
        const T *input, T *output, T _alpha_name, const dims_t &in_shape,      \
        const strides_t &input_strides, const dims_t &out_shape,               \
        const strides_t &out_strides,                                          \
        NNCASE_UNUSED kernel_context &context) noexcept {                      \
        return apply(out_shape, [&](const dims_t &index) -> result<void> {     \
            const auto in_index =                                              \
                kernels::detail::get_reduced_offset(index, in_shape);          \
            auto src_idx = offset(input_strides, in_index);                    \
            auto dst_idx = offset(out_strides, in_index);                      \
            auto x = input[src_idx];                                           \
            output[dst_idx] = _compute;                                        \
            return ok();                                                       \
        });                                                                    \
    }

#define FLOAT_UNARY_WITH_MUL_OP_TEMPLATE(_name, _alpha_name)                   \
    result<void> _name##_impl(                                                 \
        const float *input, float *output, const dims_t &input_shape,          \
        const strides_t &input_strides, const dims_t &out_shape,               \
        const strides_t &out_strides, NNCASE_UNUSED kernel_context &context);  \
    result<value_t> nncase::kernels::stackvm::_name(                           \
        value_t input, value_t _alpha_name, value_t output,                    \
        kernel_context &context) {                                             \
        try_f32_input(input_mem, input);                                       \
        try_to_scalar(_alpha_name##_value, _alpha_name, float);                \
        auto dtype = input_tensor->dtype();                                    \
        try_f32_output(out_mem, output, dtype, input_tensor->shape());         \
        try_(_name##_impl(input_mem, out_mem, _alpha_name##_value,             \
                          input_tensor->shape(), input_tensor->strides(),      \
                          output_tensor->shape(), output_tensor->strides(),    \
                          context));                                           \
        return ok(output);                                                     \
    }

// _alpha_name is a var used in kernel
#define FLOAT_UNARY_WITH_MUL_TEMPLATE(_name, _alpha_name, _compute)            \
    FLOAT_UNARY_WITH_MUL_IMPL_TEMPLATE(_name, _alpha_name, _compute)           \
    FLOAT_UNARY_WITH_MUL_OP_TEMPLATE(_name, _alpha_name)