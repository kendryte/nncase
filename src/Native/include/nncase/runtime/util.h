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
#include "runtime_tensor.h"
#include "simple_types.h"
#include <nncase/api.h>
#include <nncase/runtime/runtime_op_utility.h>

BEGIN_NS_NNCASE_RUNTIME

// cast macro
#define IN_CAST(_ty, _name) reinterpret_cast<const _ty *>(_name)
#define OUT_CAST(_ty, _name) reinterpret_cast<_ty *>(_name)
#define SCALAR_CAST(_ty, _name) *reinterpret_cast<const _ty *>(_name)
#define IN_BYTE_CAST(_var) IN_CAST(gsl::byte, _var)
#define OUT_BYTE_CAST(_var) OUT_CAST(gsl::byte, _var)

// compare type
// for typecode, datatype_t, tensor(tensor->dtype())
inline result<bool> cmp_dt_impl(datatype_t lhs, datatype_t rhs) {
    try_var(l, to_typecode(lhs));
    try_var(r, to_typecode(rhs));
    return ok(l == r);
}

inline bool cmp_dt(tensor lhs, tensor rhs) {
    auto result = cmp_dt_impl(lhs->dtype(), rhs->dtype());
    return result.is_ok() && result.unwrap();
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

// tuple helper
template <typename F>
inline result<void> tuple_for_each_with_i(tuple inputs, F &&f) {
    for (size_t i = 0; i < inputs->fields().size(); ++i) {
        try_(f(inputs->fields()[i], i));
    }
    return ok();
}

// todo:not process nest tuple
template <typename T, bool IsResult, typename F>
inline result<std::vector<T>> get_from_tuple_with_result(tuple inputs, F &&f) {
    std::vector<T> data(inputs->fields().size());
    for (size_t i = 0; i < inputs->fields().size(); ++i) {
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

// get input and output
inline result<void> alloc_output(value_t &output, datatype_t dtype,
                                 gsl::span<const size_t> out_shape) {
    // TODO: copy back output
    if (output.empty()) {
        try_var(typecode, to_typecode(dtype));
        try_var(out_tensor, hrt::create(typecode, dims_t(out_shape)));
        output = out_tensor.impl();
    } else {
        try_var(
            out_tensor,
            output.as<tensor>()) if (out_tensor->shape() !=
                                     out_shape) return err(nncase_errc::
                                                               shape_mismatch);
    }
    return ok();
}

inline result<void> check_tuple_shape(value_t &outputs,
                                      const std::vector<dims_t> &out_shapes) {
    try_var(output_tuple, outputs.as<tuple>());
    try_(tuple_for_each_with_i(
        output_tuple, [&](auto &output, auto i) -> result<void> {
            try_var(out_tensor, output.template as<tensor>());
            if (out_tensor->shape() != gsl::span(out_shapes[i])) {
                return err(nncase_errc::shape_mismatch);
            } else {
                return ok();
            }
        }));
    return ok();
}

inline result<void> alloc_tuple_output(value_t &outputs,
                                       const std::vector<datatype_t> dtypes,
                                       const std::vector<dims_t> &out_shapes) {
    if (outputs.empty()) {
        auto size = out_shapes.size();
        std::vector<value_t> fields(size);
        for (size_t i = 0; i < size; ++i) {
            auto output = value_t();
            try_(alloc_output(output, dtypes[i], out_shapes[i]));
            fields[i] = output;
        }
        outputs = tuple(std::in_place, std::move(fields));
    } else {
        try_(check_tuple_shape(outputs, out_shapes));
    }
    return ok();
}

inline result<void> alloc_output(value_t &outputs, datatype_t dtype,
                                 const std::vector<dims_t> &out_shapes) {
    if (outputs.empty()) {
        auto size = out_shapes.size();
        std::vector<value_t> fields(size);
        for (size_t i = 0; i < size; ++i) {
            auto output = value_t();
            try_(alloc_output(output, dtype, out_shapes[i]));
            fields[i] = output;
        }
        outputs = tuple(std::in_place, std::move(fields));
    } else {
        try_(check_tuple_shape(outputs, out_shapes));
    }
    return ok();
}

inline result<host_buffer_slice> get_host_buffer(tensor tensor) {
    try_var(tensor_host, tensor->to_host());
    try_var(tensor_buffer, tensor_host->buffer().as_host());
    return ok(tensor_buffer);
}

inline result<gsl::span<gsl::byte>> get_output_span(tensor output) {
    try_var(output_buffer, get_host_buffer(output));
    try_var(output_map, output_buffer.map(map_write));
    return ok(output_map.buffer());
}

inline result<gsl::byte *> get_output_data(tensor output) {
    try_var(output_buffer, get_output_span(output));
    return ok(output_buffer.data());
}

inline result<std::vector<gsl::byte *>> get_output_data(tuple outputs) {
    return get_from_tuple_with_result<gsl::byte *, true>(
        outputs, [](tensor &input) { return get_output_data(input); });
}

inline result<gsl::span<gsl::byte>> get_input_span(tensor input) {
    try_var(input_buffer, get_host_buffer(input));
    try_var(input_map, input_buffer.map(map_read));
    return ok(input_map.buffer());
}

inline result<gsl::byte *> get_input_data(tensor input) {
    try_var(input_buffer, get_input_span(input));
    return ok(input_buffer.data());
}

inline result<std::vector<gsl::byte *>> get_input_data(tuple inputs) {
    return get_from_tuple_with_result<gsl::byte *, true>(
        inputs, [](tensor &input) { return get_input_data(input); });
}

inline result<std::vector<gsl::byte *>> get_readonly_span(tuple inputs) {
    return get_input_data(inputs);
}

inline result<gsl::byte *> get_readonly_span(tensor input) {
    return get_input_data(input);
}

// some macro about get value for tensor_ops.cpp
// implicit define tensor/tuple for try_input[xxx] and try_output[xxx]
// e.g. try_input(in_mem, input) ->
// 1. in_mem: const gsl::byte*
// 2. input_tensor: tensor
#define try_alloc_output(_out_tensor, _dt, _shape, _is_tuple)                  \
    try_(alloc_output(_out_tensor, _dt, _shape));

#define try_input_impl(_var_name, _value_name, _value_kind)                    \
    try_var(_value_name##_##_value_kind, _value_name.as<_value_kind>());       \
    try_var(_var_name, get_input_data(_value_name##_##_value_kind))

#define try_input(_var_name, _value_name)                                      \
    try_input_impl(_var_name, _value_name, tensor)
#define try_tuple_input(_var_name, _value_name)                                \
    try_input_impl(_var_name, _value_name, tuple)

#define try_input_with_value_type(_var_name, _value_name, _ty)                 \
    try_input(__##_var_name, _value_name);                                     \
    auto *_var_name = IN_CAST(_ty, __##_var_name);

#define try_tuple_field0(_input0_name, _tuple_name)                            \
    try_var(_input0_name, _tuple_name->fields()[0].as<tensor>());

#define try_input_with_ty(_var_name, _value_name, _ty)                         \
    try_input(__##_var_name, _value_name);                                     \
    try_(type_only_check<_ty>(_value_name##_tensor));                          \
    auto _var_name = reinterpret_cast<const _ty *>(__##_var_name)

#define try_integer_input(_var_name, _value_name)                              \
    try_input_with_ty(_var_name, _value_name, int64_t)

#define try_f32_input(_var_name, _value_name)                                  \
    try_input_with_ty(_var_name, _value_name, float)
#define try_f32_output(_var_name, _value_name, _out_shape)                     \
    try_output(__##_var_name, _value_name, dt_float32, _out_shape);            \
    auto _var_name = reinterpret_cast<float *>(__##_var_name)

// todo:when _value_kind is tuple, _value_name_tensor is a bad name
#define try_output_impl(_var_name, _value_name, _dt, _out_shape, _value_kind,  \
                        _is_tuple)                                             \
    try_alloc_output(_value_name, _dt, _out_shape, _is_tuple);                 \
    try_var(_value_name##_##_value_kind, _value_name.as<_value_kind>());       \
    try_var(_var_name, get_output_data(_value_name##_##_value_kind))

#define try_output(_var_name, _value_name, _dt, _out_shape)                    \
    try_output_impl(_var_name, _value_name, _dt, _out_shape, tensor, false)

#define try_output_like_input(_var_name, _value_name, _tensor)                 \
    try_output(_var_name, _value_name, (_tensor)->dtype(), (_tensor)->shape())

#define try_tuple_output(_var_name, _value_name, _dt, _out_shapes)             \
    try_output_impl(_var_name, _value_name, _dt, _out_shapes, tuple, true)

#define try_value_as(_var_name, _value_name, f_name)                           \
    try_var(_var_name, value_as_##f_name(_value_name))
#define try_strides(_var_name, _value_name)                                    \
    try_value_as(_var_name, _value_name, strides)
#define try_dims(_var_name, _value_name)                                       \
    try_value_as(_var_name, _value_name, dims)
#define try_positive_axes(_var_name, _value_name, _rank)                       \
    try_var(_var_name, value_as_positive_axes(_value_name, _rank))

#define try_axes(_var_name, _value_name)                                       \
    try_var(_var_name, value_as_axes(_value_name))
#define try_paddings(_var_name, _value_name)                                   \
    try_value_as(_var_name, _value_name, paddings)
#define try_value_as_t(_var_name, _value_name, _ty, f_name)                    \
    try_var(_var_name, value_as_##f_name<_ty>(_value_name))
#define try_to_scalar(_var_name, _value_name, _ty)                             \
    try_var(_var_name, value_to_scalar<_ty>(_value_name))

#define try_float_scalar(_var_name, _value_name)                               \
    try_to_scalar(_var_name, _value_name, float)

#define try_to_integer(_var_name, _value_name)                                 \
    try_to_scalar(_var_name, _value_name, int64_t)

#define try_positive_axis_with_rank(_var_name, _value_name, _rank)             \
    try_to_scalar(__##_var_name, _value_name, int64_t);                        \
    auto _var_name = positive_index(__##_var_name, _rank)

#define try_positive_axis(_var_name, _value_name, _input_tensor)               \
    try_positive_axis_with_rank(_var_name, _value_name,                        \
                                _input_tensor->shape().size())

#define try_typecode(_var_name, _tensor_name)                                  \
    try_var(_var_name, to_typecode(_tensor_name->dtype()))

#define try_ref(op, ...) try_(reference::op(__VA_ARGS__))

// implicit set var name
#define try_out_mem(_value_name, _dt, _out_shape)                              \
    try_output(_value_name##_mem, _value_name, _dt, _out_shape)
#define try_f32_out_mem(_value_name, _out_shape)                               \
    try_f32_output(_value_name##_mem, _value_name, _out_shape)

#define try_in_mem(_value_name) try_input(_value_name##_mem, _value_name)
#define try_f32_in_mem(_value_name)                                            \
    try_f32_input(_value_name##_mem, _value_name)

#define try_float_scalar_v(_value_name) try_to_scalar_v(_value_name, float)

#define try_to_scalar_v(_value_name, _ty)                                      \
    try_to_scalar(_value_name##_value, _value_name, _ty)

#define try_integer_v(_value_name)                                             \
    try_to_integer(_value_name##_value, _value_name)

#define try_dims_v(_value_name) try_dims(_value_name##_value, _value_name)

// other cast macro
#define to_tensor(_tensor_name, _value)                                        \
    try_var(_tensor_name, _value.as<tensor>());

#define to_tensor_t(_value) to_tensor(_value##_tensor, _value)

#define KERNEL_FINISH return ok(output)
#define TUPLE_FINISH return ok(output_tuple)

// get data from value
template <typename TI, typename TO>
itlib::small_vector<TO, 8> to_vec(const gsl::byte *input, size_t size) {
    auto in_ptr = reinterpret_cast<const TI *>(input);
    auto vec = itlib::small_vector<TO, 8>(size);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = (TO)in_ptr[i];
    }
    return vec;
}

#define RETURN_RESULT_SELECT(RETURN_RESULT_IMPL)                               \
    RETURN_RESULT_IMPL(bool);                                                  \
    RETURN_RESULT_IMPL(int8_t);                                                \
    RETURN_RESULT_IMPL(uint8_t);                                               \
    RETURN_RESULT_IMPL(int32_t);                                               \
    RETURN_RESULT_IMPL(uint32_t);                                              \
    RETURN_RESULT_IMPL(int64_t);                                               \
    RETURN_RESULT_IMPL(uint64_t);                                              \
    RETURN_RESULT_IMPL(float);                                                 \
    RETURN_RESULT_IMPL(double);

template <typename T>
inline result<T> value_to_scalar([[maybe_unused]] value_t value) {
    try_input(input, value);
    // todo: maybe this is a bad way?
#define RETURN_RESULT(_in_type)                                                \
    if (cmp_type<_in_type>(value_tensor->dtype())) {                           \
        return ok((T)(*reinterpret_cast<const _in_type *>(input)));            \
    }
    RETURN_RESULT_SELECT(RETURN_RESULT);
    return err(nncase_errc::datatype_mismatch);
#undef RETURN_RESULT
}

template <typename T>
inline result<itlib::small_vector<T, 8>> value_as_Ts(value_t value) {
    try_input(input, value);
    assert(value_tensor->shape().size() <= 1);
    auto size =
        value_tensor->shape().size() == 0 ? 1 : value_tensor->shape()[0];
#define RETURN_RESULT(_in_type)                                                \
    if (cmp_type<_in_type>(value_tensor->dtype())) {                           \
        return ok(to_vec<_in_type, T>(input, size));                           \
    }

    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, int64_t> || std::is_same_v<T, size_t>,
                  "not suppported type");
    RETURN_RESULT(int32_t);
    RETURN_RESULT(uint32_t);
    RETURN_RESULT(int64_t);
    RETURN_RESULT(uint64_t);
#undef RETURN_RESULT
    return err(nncase_errc::datatype_mismatch);
}

inline result<dims_t> value_as_dims(value_t value) {
    return value_as_Ts<dims_t::value_type>(value);
}

inline result<axes_t> value_as_axes(value_t value) {
    return value_as_Ts<axes_t::value_type>(value);
}

inline size_t positive_index(int index, size_t rank) {
    return index < 0 ? index + rank : index;
}

// todo:refactor, same as axes but should positive
inline result<dims_t> value_as_positive_axes(value_t value, size_t rank) {
    try_input(input, value);
    assert(value_tensor->shape().size() == 1);
    auto size = value_tensor->shape()[0];
    auto axis = dims_t(size);
    for (size_t i = 0; i < size; ++i) {
        if (cmp_type<int32_t>(value_tensor->dtype())) {
            axis[i] = (dims_t::value_type)positive_index(
                IN_CAST(int32_t, input)[i], rank);
        } else if (cmp_type<int64_t>(value_tensor->dtype())) {
            axis[i] = (dims_t::value_type)positive_index(
                IN_CAST(int64_t, input)[i], rank);
        } else {
            return err(nncase_errc::datatype_mismatch);
        }
    }
    return ok(axis);
}

inline result<strides_t> value_as_strides(value_t value) {
    return value_as_Ts<strides_t::value_type>(value);
}

inline size_t compute_size(tensor t) {
    return compute_size(t->shape(), t->strides());
}

inline result<paddings_t> value_as_paddings([[maybe_unused]] value_t value) {
    try_input(input, value);
    auto size = compute_size(value_tensor);
    auto dims = size / 2;
    auto pads = paddings_t(dims);
    auto dt = value_tensor->dtype();
    for (size_t i = 0; i < dims; ++i) {
        if (cmp_type<int32_t>(dt)) {
            pads[i].before = *(IN_CAST(int32_t, input) + 2 * i);
            pads[i].after = *(IN_CAST(int32_t, input) + 2 * i + 1);
            pads[i].interior = 0;
        } else if (cmp_type<int64_t>(dt)) {
            pads[i].before = *(IN_CAST(int64_t, input) + 2 * i);
            pads[i].after = *(IN_CAST(int64_t, input) + 2 * i + 1);
            pads[i].interior = 0;
        } else {
            return err(nncase_errc::datatype_mismatch);
        }
    }
    return ok(pads);
}

// kernel util
inline bool is_contiguous(tensor tensor) {
    return is_contiguous(tensor->shape(), tensor->strides());
}

#define not_impl_no_contiguous(tensor)                                         \
    if (!is_contiguous(tensor)) {                                              \
        return err(nncase_errc::shape_mismatch);                               \
    }

#define TYPE_SELECT(_typecode, _impl)                                          \
    switch (_typecode) {                                                       \
    case dt_float32:                                                           \
        _impl(float);                                                          \
    case dt_float16:                                                           \
        _impl(half);                                                           \
    case dt_bfloat16:                                                          \
        _impl(bfloat16);                                                       \
    case dt_int8:                                                              \
        _impl(int8_t);                                                         \
    case dt_int16:                                                             \
        _impl(int16_t);                                                        \
    case dt_int32:                                                             \
        _impl(int32_t);                                                        \
    case dt_int64:                                                             \
        _impl(int64_t);                                                        \
    case dt_uint8:                                                             \
        _impl(uint8_t);                                                        \
    case dt_uint16:                                                            \
        _impl(uint16_t);                                                       \
    case dt_uint32:                                                            \
        _impl(uint32_t);                                                       \
    case dt_uint64:                                                            \
        _impl(uint64_t);                                                       \
    case dt_float64:                                                           \
        _impl(double);                                                         \
    case dt_boolean:                                                           \
        _impl(bool);                                                           \
    default:                                                                   \
        return err(std::errc::not_supported);                                  \
    }

// kernel dispatch for single input
#define CONTIGUOUS_KERNEL(_op, _in_tensor, ...)                                \
    if (is_contiguous(_in_tensor)) {                                           \
        try_(optimized::_op(__VA_ARGS__))                                      \
    } else {                                                                   \
        try_(reference::_op(__VA_ARGS__))                                      \
    }

// used for op only do reshape
inline tensor tensor_reshape(tensor in_tensor,
                             gsl::span<const size_t> new_shape) {
    auto strides = get_default_strides(new_shape);
    return tensor(std::in_place, in_tensor->dtype(), new_shape, strides,
                  in_tensor->buffer());
}

inline bool is_scalar(tensor t) noexcept { return t->shape().empty(); }
inline bool is_scalar(gsl::span<const size_t> t) noexcept { return t.empty(); }

template <typename F>
inline result<void> integer_cast(datatype_t type, const gsl::byte *input,
                                 F &&f) {
    if (cmp_type<int32_t>(type)) {
        try_(f(IN_CAST(int32_t, input)));
    } else if (cmp_type<int64_t>(type)) {
        try_(f(IN_CAST(int64_t, input)));
    } else {
        return err(nncase_errc::datatype_mismatch);
    }
    return ok();
}

// used for slice args
inline std::tuple<axes_t, axes_t, axes_t>
slice_fill(gsl::span<const size_t> in_shape, axes_t &begins_value,
           axes_t &ends_value, axes_t &strides_value, axes_t axes_value) {
    auto ndim = in_shape.size();
    axes_t begin_values(ndim, 0);
    axes_t end_values(in_shape.begin(), in_shape.end());
    axes_t strides_values(ndim, 1);
    for (size_t i = 0; i < ndim; ++i) {
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

inline dims_t to_4d(dims_t in_a_shape) {
    auto size = 4 - in_a_shape.size();
    for (size_t i = 0; i < size; ++i) {
        in_a_shape.insert(in_a_shape.begin(), 1);
    }
    return in_a_shape;
}

inline void shrink_memory_pool() {
    buffer_allocator::host().shrink_memory_pool();
}

END_NS_NNCASE_RUNTIME