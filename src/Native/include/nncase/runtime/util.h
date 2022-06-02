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
#include "../tensor.h"
#include "allocator.h"
#include "buffer.h"
#include "error.h"
#include "host_buffer.h"
#include "simple_types.h"
#include <nncase/api.h>
#include <nncase/runtime/runtime_op_utility.h>

namespace nncase::runtime {
inline bool is_scalar(tensor t) noexcept { return t->shape().empty(); }

template <typename T>
inline result<T> value_to_scalar([[maybe_unused]] value_t v) {
    throw "NotImplement";
}

inline result<scalar> tensor_as_scalar([[maybe_unused]] value_t v) {
    throw "NotImplement";
}

inline result<dims_t> value_as_dims([[maybe_unused]] value_t v) {
    throw "NotImplement";
}

inline result<strides_t> value_as_strides([[maybe_unused]] value_t v) {
    throw "NotImplement";
}

inline result<paddings_t> value_as_paddings([[maybe_unused]] value_t v) {
    throw "NotImplement";
}

inline result<quant_param_t> value_as_quant_param([[maybe_unused]] value_t v) {
    throw "NotImplement";
}

template <typename T>
inline result<T *> value_as_array([[maybe_unused]] value_t v) {
    throw "NotImplement";
}

template <typename T, bool IsResult, typename F>
inline result<std::vector<T>> get_from_tuple_with_result(tuple inputs, F &&f) {
    std::vector<T> data(inputs->fields().size());
    for (int i = 0; i < inputs->fields().size(); ++i) {
        try_var(input, inputs->fields()[i].as<tensor>());
        if constexpr (IsResult) {
            try_var(in, f(input));
            data[i] = in;
        } else {
            data[i] = f(input);
        }
    }
    return ok(data);
}

template <typename T, typename F>
inline result<std::vector<T>> get_from_tuple(tuple inputs, F &&f) {
    return get_from_tuple_with_result<T, false>(inputs, f);
}

inline result<std::vector<dims_t>> get_shapes(tuple inputs) {
    return get_from_tuple<dims_t>(inputs,
                                  [](auto &input) { return input->shape(); });
}

inline result<std::vector<dims_t>> get_strides(tuple inputs) {
    return get_from_tuple<dims_t>(inputs,
                                  [](auto &input) { return input->strides(); });
}

inline result<void> alloc_output(value_t output_v, datatype_t dtype,
                                 const dims_t &out_shape) {
    // TODO: copy back output
    try_var(output, output_v.as<tensor>());
    assert(output.empty());
    if (output.empty()) {
        auto out_strides = get_default_strides(out_shape);
        try_var(out_buffer, buffer_allocator::host().allocate(
                                get_bytes(dtype, out_shape, out_strides), {}));
        output =
            tensor(std::in_place, dtype, out_shape, out_strides, out_buffer);
    } else {
        if (output->shape() != out_shape)
            return err(nncase_errc::shape_mismatch);
    }
    return ok();
}

inline result<host_buffer_slice> get_host_buffer(tensor tensor) {
    try_var(tensor_host, tensor->to_host());
    try_var(tensor_buffer, tensor_host->buffer().as_host());
    return ok(tensor_buffer);
}

inline result<gsl::byte *> get_output_data(tensor output) {
    try_var(output_buffer, get_host_buffer(output));
    try_var(output_map, output_buffer.map(map_write));
    return ok(output_map.buffer().data());
}

inline result<gsl::byte *> get_input_data(tensor input) {
    try_var(input_buffer, get_host_buffer(input));
    try_var(input_map, input_buffer.map(map_read));
    return ok(input_map.buffer().data());
}

inline result<std::vector<gsl::byte *>> get_input_data(tuple inputs) {
    return get_from_tuple_with_result<gsl::byte *, true>(
        inputs, [](tensor &input) { return get_input_data(input); });
}

inline result<bool> cmp_dt_impl(datatype_t lhs, datatype_t rhs) {
    try_var(l, to_typecode(lhs));
    try_var(r, to_typecode(rhs));
    return ok(l == r);
}

inline bool cmp_dt(datatype_t lhs, datatype_t rhs) {
    auto result = cmp_dt_impl(lhs, rhs);
    return result.is_ok() && result.unwrap();
}

template <typename T> inline bool cmp_type(datatype_t dt) {
    return cmp_dt(datatype_t::from_type<T>(), dt);
}

template <typename T> inline result<bool> type_only_check(tensor input) {
    return cmp_dt_impl(input->dtype(), datatype_t::from_type<T>());
}

inline result<bool> float_only_check(tensor input) {
    return cmp_dt_impl(input->dtype(), datatype_t::float32);
}

inline int positive_index(int index, int rank) {
    return index < 0 ? index + rank : index;
}

#define try_alloc_output(out_mem, _out_tensor, _dt, _shape)                    \
    try_(alloc_output(_out_tensor, _dt, _shape));
#define try_check_float(_tensor_name) try_(float_only_check(_tensor_name))

#define try_input_impl(_var_name, _value_name, _value_kind)                    \
    try_var(_value_name##_tensor, _value_name.as<_value_kind>());              \
    try_var(_var_name, get_input_data(_value_name##_tensor))

#define try_input(_var_name, _value_name)                                      \
    try_input_impl(_var_name, _value_name, tensor)
#define try_tuple_input(_var_name, _value_name)                                \
    try_input_impl(_var_name, _value_name, tuple)

#define try_input_with_ty(_var_name, _value_name, _ty)                         \
    try_input(__##_var_name, _value_name)                                      \
        try_(type_only_check<_ty>(_value_name##_tensor)) auto _var_name =      \
            reinterpret_cast<const _ty *>(__##_var_name)

#define try_f32_input(_var_name, _tensor_name)                                 \
    try_input_with_ty(_var_name, _tensor_name, float)
#define try_f32_output(_var_name, _tensor_name, _dt, _out_shape)               \
    try_output(__##_var_name, _tensor_name, _dt, _out_shape);                  \
    auto _var_name = reinterpret_cast<float *>(__##_var_name)

#define try_output_impl(_var_name, _value_name, _dt, _out_shape, _value_kind)  \
    try_var(_value_name##_tensor, _value_name.as<_value_kind>());              \
    try_alloc_output(out_mem, _value_name, _dt, _out_shape);                   \
    try_var(_var_name, get_output_data(_value_name##_tensor))

#define try_output(_var_name, _value_name, _dt, _out_shape)                    \
    try_output_impl(_var_name, _value_name, _dt, _out_shape, tensor)

#define try_tuple_output(_var_name, _value_name, _dt, _out_shape)              \
    try_output_impl(_var_name, _value_name, _dt, _out_shape, tuple)

#define try_value_as(_var_name, _tensor_name, f_name)                          \
    try_var(_var_name, value_as_##f_name(_tensor_name))
#define try_strides(_var_name, _tensor_name)                                   \
    try_value_as(_var_name, _tensor_name, strides)
#define try_dims(_var_name, _tensor_name)                                      \
    try_value_as(_var_name, _tensor_name, dims)
#define try_paddings(_var_name, _tensor_name)                                  \
    try_value_as(_var_name, _tensor_name, paddings)
#define try_value_as_t(_var_name, _tensor_name, _ty, f_name)                   \
    try_var(_var_name, value_as_##f_name<_ty>(_tensor_name))
#define try_to_scalar(_var_name, _tensor_name, _ty)                            \
    try_var(_var_name, value_to_scalar<_ty>(_tensor_name))
#define try_array(_var_name, _tensor_name, _ty)                                \
    try_value_as_t(_var_name, _tensor_name, _ty, array)

} // namespace nncase::runtime