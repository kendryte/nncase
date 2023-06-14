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

#include <nncase/runtime/util.h>

#define FLOAT_UNARY_IMPL_TEMPLATE(_name, _compute)                             \
    template <class T>                                                         \
    result<void> _name##_impl(                                                 \
        const T *input, T *output, gsl::span<const size_t> in_shape,                     \
        const strides_t &input_strides, gsl::span<const size_t> out_shape,               \
        const strides_t &out_strides,                                          \
        NNCASE_UNUSED kernel_context &context) noexcept {                      \
        return apply(out_shape, [&](gsl::span<const size_t> index) -> result<void> {     \
            const auto in_index =                                              \
                kernels::detail::get_reduced_offset(index, in_shape);          \
            auto src_idx = offset(input_strides, in_index);                    \
            auto dst_idx = offset(out_strides, in_index);                      \
            auto x = input[src_idx];                                           \
            output[dst_idx] = _compute;                                        \
            return ok();                                                       \
        });                                                                    \
    }                                                                          \
    template <class T>                                                         \
    result<void> _name##_opt_impl(                                             \
        const T *input, T *output, gsl::span<const size_t> in_shape,                     \
        [[maybe_unused]] const strides_t &input_strides,                       \
        [[maybe_unused]] gsl::span<const size_t> out_shape,                              \
        [[maybe_unused]] const strides_t &out_strides,                         \
        NNCASE_UNUSED kernel_context &context) noexcept {                      \
        for (int i = 0; i < compute_size(in_shape); ++i) {                     \
            auto x = input[i];                                                 \
            output[i] = _compute;                                              \
        }                                                                      \
        return ok();                                                           \
    }

#define FLOAT_UNARY_OP_TEMPLATE(_name)                                         \
    result<value_t> nncase::kernels::stackvm::_name(                           \
        value_t input, value_t output, kernel_context &context) {              \
        try_f32_input(input_mem, input);                                       \
        auto dtype = input_tensor->dtype();                                    \
        try_f32_output(out_mem, output, input_tensor->shape());                \
        if (is_contiguous(input_tensor)) {                                     \
            try_(_name##_opt_impl(input_mem, out_mem, input_tensor->shape(),   \
                                  input_tensor->strides(),                     \
                                  output_tensor->shape(),                      \
                                  output_tensor->strides(), context));         \
        } else {                                                               \
            try_(_name##_impl(input_mem, out_mem, input_tensor->shape(),       \
                              input_tensor->strides(), output_tensor->shape(), \
                              output_tensor->strides(), context));             \
        }                                                                      \
        return ok(output);                                                     \
    }

#define FLOAT_UNARY_TEMPLATE(_name, _compute)                                  \
    FLOAT_UNARY_IMPL_TEMPLATE(_name, _compute)                                 \
    FLOAT_UNARY_OP_TEMPLATE(_name)

#define FLOAT_UNARY_WITH_MUL_IMPL_TEMPLATE(_name, _alpha_name, _compute)       \
    template <class T>                                                         \
    result<void> _name##_impl(                                                 \
        const T *input, T *output, T _alpha_name, gsl::span<const size_t> in_shape,      \
        const strides_t &input_strides, gsl::span<const size_t> out_shape,               \
        const strides_t &out_strides,                                          \
        NNCASE_UNUSED kernel_context &context) noexcept {                      \
        return apply(out_shape, [&](gsl::span<const size_t> index) -> result<void> {     \
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
        const float *input, float *output, gsl::span<const size_t> input_shape,          \
        const strides_t &input_strides, gsl::span<const size_t> out_shape,               \
        const strides_t &out_strides, NNCASE_UNUSED kernel_context &context);  \
    result<value_t> nncase::kernels::stackvm::_name(                           \
        value_t input, value_t _alpha_name, value_t output,                    \
        kernel_context &context) {                                             \
        try_f32_input(input_mem, input);                                       \
        try_to_scalar(_alpha_name##_value, _alpha_name, float);                \
        auto dtype = input_tensor->dtype();                                    \
        try_f32_output(out_mem, output, input_tensor->shape());                \
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

#define MKFNS(fn, ...)                                                         \
    MKFN_N(fn, ##__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)(__VA_ARGS__)
#define MKFN_N(fn, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n, ...) fn##n

#define FLOAT_ARGS_EXPAND(...) MKFNS(FLOAT_ARGS_EXPAND, ##__VA_ARGS__)

#define FLOAT_ARG(_a) [[maybe_unused]] float _a
#define FLOAT_ARGS_EXPAND0(_a) FLOAT_ARG(_a)
#define FLOAT_ARGS_EXPAND1(_a, _b) FLOAT_ARG(_a), FLOAT_ARG(_b)
#define FLOAT_ARGS_EXPAND2(_a, _b, _c)                                         \
    FLOAT_ARG(_a), FLOAT_ARG(_b), FLOAT_ARG(_c)
#define FLOAT_ARGS_EXPAND3(_a, _b, _c, _d)                                     \
    FLOAT_ARG(_a), FLOAT_ARG(_b), FLOAT_ARG(_c), FLOAT_ARG(_d)
#define FLOAT_ARGS_EXPAND4(_a, _b, _c, _d, _e)                                 \
    FLOAT_ARG(_a), FLOAT_ARG(_b), FLOAT_ARG(_c), FLOAT_ARG(_d), FLOAT_ARG(_e)
#define FLOAT_ARGS_EXPAND5(_a, _b, _c, _d, _e, _f)                             \
    FLOAT_ARG(_a), FLOAT_ARG(_b), FLOAT_ARG(_c), FLOAT_ARG(_d), FLOAT_ARG(_e), \
        FLOAT_ARG(_f)

#define FLOAT_ACTIVATION_IMPL_TEMPLATE(_name, _compute, ...)                   \
    template <class T>                                                         \
    result<void> _name##_impl(                                                 \
        const T *input, T *output, gsl::span<const size_t> in_shape,                     \
        const strides_t &input_strides, gsl::span<const size_t> out_shape,               \
        const strides_t &out_strides, FLOAT_ARGS_EXPAND(__VA_ARGS__),          \
        NNCASE_UNUSED kernel_context &context) noexcept {                      \
        return apply(out_shape, [&](gsl::span<const size_t> index) -> result<void> {     \
            const auto in_index =                                              \
                kernels::detail::get_reduced_offset(index, in_shape);          \
            auto src_idx = offset(input_strides, in_index);                    \
            auto dst_idx = offset(out_strides, in_index);                      \
            auto x = input[src_idx];                                           \
            output[dst_idx] = _compute;                                        \
            return ok();                                                       \
        });                                                                    \
    }

#define VALUE_ARGS_EXPAND(...) MKFNS(VALUE_ARGS_EXPAND, ##__VA_ARGS__)

#define VALUE_ARGS_EXPAND0(_a) value_t _a
#define VALUE_ARGS_EXPAND1(_a, _b) value_t _a, value_t _b
#define VALUE_ARGS_EXPAND2(_a, _b, _c) value_t _a, value_t _b, value_t _c
#define VALUE_ARGS_EXPAND3(_a, _b, _c, _d)                                     \
    value_t _a, value_t _b, value_t _c, value_t _d
#define VALUE_ARGS_EXPAND4(_a, _b, _c, _d, _e)                                 \
    value_t _a, value_t _b, value_t _c, value_t _d, value_t _e
#define VALUE_ARGS_EXPAND5(_a, _b, _c, _d, _e, _f)                             \
    value_t _a, value_t _b, value_t _c, value_t _d, value_t _e, value_t _f

#define FLOAT_SCALAR(_var) try_float_scalar(_var##_value, _var);
#define READ_FLOAT_SCALAR_EXPAND(...)                                          \
    MKFNS(READ_FLOAT_SCALAR_EXPAND, ##__VA_ARGS__)
#define READ_FLOAT_SCALAR_EXPAND0(_a) FLOAT_SCALAR(_a)
#define READ_FLOAT_SCALAR_EXPAND1(_a, _b)                                      \
    FLOAT_SCALAR(_a);                                                          \
    FLOAT_SCALAR(_b)
#define READ_FLOAT_SCALAR_EXPAND2(_a, _b, _c)                                  \
    FLOAT_SCALAR(_a);                                                          \
    FLOAT_SCALAR(_b);                                                          \
    FLOAT_SCALAR(_c)
#define READ_FLOAT_SCALAR_EXPAND5(_a, _b, _c, _d, _e, _f)                      \
    FLOAT_SCALAR(_a);                                                          \
    FLOAT_SCALAR(_b);                                                          \
    FLOAT_SCALAR(_c);                                                          \
    FLOAT_SCALAR(_d);                                                          \
    FLOAT_SCALAR(_e);                                                          \
    FLOAT_SCALAR(_f)

#define SCALAR_VALUE_EXPAND(...) MKFNS(SCALAR_VALUE, ##__VA_ARGS__)
#define SCALAR_VALUE0(_a) _a##_value
#define SCALAR_VALUE1(_a, _b) _a##_value, _b##_value
#define SCALAR_VALUE2(_a, _b, _c) _a##_value, _b##_value, _c##_value
#define SCALAR_VALUE3(_a, _b, _c, _d)                                          \
    _a##_value, _b##_value, _c##_value, _d##_value
#define SCALAR_VALUE4(_a, _b, _c, _d, _e)                                      \
    _a##_value, _b##_value, _c##_value, _d##_value, _e##_value
#define SCALAR_VALUE5(_a, _b, _c, _d, _e, _f)                                  \
    _a##_value, _b##_value, _c##_value, _d##_value, _e##_value, _f##_value

#define FLOAT_ACTIVATION_OP_TEMPLATE(_name, ...)                               \
    result<value_t> nncase::kernels::stackvm::_name(                           \
        value_t input, VALUE_ARGS_EXPAND(__VA_ARGS__), value_t output,         \
        kernel_context &context) {                                             \
        try_f32_input(input_mem, input);                                       \
        auto dtype = input_tensor->dtype();                                    \
        READ_FLOAT_SCALAR_EXPAND(__VA_ARGS__);                                 \
        try_f32_output(out_mem, output, input_tensor->shape());                \
        try_(_name##_impl(input_mem, out_mem, input_tensor->shape(),           \
                          input_tensor->strides(), output_tensor->shape(),     \
                          output_tensor->strides(),                            \
                          SCALAR_VALUE_EXPAND(__VA_ARGS__), context));         \
        return ok(output);                                                     \
    }

#define FLOAT_ACTIVATION_TEMPLATE(_name, _compute, ...)                        \
    FLOAT_ACTIVATION_IMPL_TEMPLATE(_name, _compute, __VA_ARGS__)               \
    FLOAT_ACTIVATION_OP_TEMPLATE(_name, __VA_ARGS__)

#define BASIC_PARAM                                                            \
    const gsl::byte *input, gsl::byte *output, gsl::span<const size_t> in_shape,         \
        gsl::span<const size_t> out_shape, const strides_t &in_strides,                  \
        const strides_t &out_strides

#define BASIC_PARAM_T                                                          \
    const T *input, T *output, gsl::span<const size_t> in_shape,                         \
        gsl::span<const size_t> out_shape, const strides_t &in_strides,                  \
        const strides_t &out_strides

#define PASS_BASIC_ARG(_input, _output)                                        \
    _input##_mem, _output##_mem, _input##_tensor->shape(),                     \
        output##_tensor->shape(), _input##_tensor->strides(),                  \
        output##_tensor->strides()

#define BASIC_BINARY_PARAM                                                     \
    const gsl::byte *lhs, const gsl::byte *rhs, gsl::byte *output,             \
        gsl::span<const size_t> lhs_shape, gsl::span<const size_t> rhs_shape,                      \
        gsl::span<const size_t> out_shape, const strides_t &lhs_strides,                 \
        const strides_t &rhs_strides, const strides_t &out_strides

#define BASIC_BINARY_PARAM_T                                                   \
    const T *lhs, const T *rhs, T *output, gsl::span<const size_t> in_shape,             \
        gsl::span<const size_t> rhs_shape, gsl::span<const size_t> out_shape,                      \
        const strides_t &lhs_strides, const strides_t &rhs_strides,            \
        const strides_t &out_strides

#define PASS_BASIC_BINARY_ARG(_lhs, _rhs, _output)                             \
    _lhs##_mem, _rhs##_mem, _output##_mem, _lhs##_tensor->shape(),             \
        _rhs##_tensor->shape(), output##_tensor->shape(),                      \
        _lhs##_tensor->strides(), _rhs##_tensor->strides(),                    \
        output##_tensor->strides()
