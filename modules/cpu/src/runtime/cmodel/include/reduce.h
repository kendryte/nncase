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

#include <apply.h>
#include <gsl/gsl-lite.hpp>
#include <runtime_utils.h>

enum class reduce_op_t : uint8_t {
    mean = 0,
    min = 1,
    max = 2,
    sum = 3,
    prod = 4,
    sum_sqr = 5,
};

namespace kernels {

namespace {
template <class T> struct identity {
    T operator()(const T &src) const noexcept { return src; }
};

template <class TShape>
size_t get_reduce_block_size(const TShape &in_shape, const TShape &axis) {
    size_t size = 1;
    for (size_t i = 0; i < in_shape.size(); i++) {
        for (size_t j = 0; j < axis.size(); j++) {
            if (i == axis[j]) {
                size *= in_shape[i];
                break;
            }
        }
    }

    return size;
}

template <class TReducer, class TPostProcess, class T>
void reduce_impl(TReducer &&reducer, TPostProcess &&post_process,
                 const T init_value, const T *input, T *output,
                 gsl::span<const size_t> in_shape, gsl::span<const size_t> axis,
                 gsl::span<const size_t> in_strides,
                 gsl::span<const size_t> out_shape,
                 gsl::span<const size_t> out_strides, bool keep_dims) noexcept {
    apply(out_shape, [&](gsl::span<const size_t> index) -> void {
        output[offset(out_strides, index)] = init_value;
    });
    apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        const auto v = input[offset(in_strides, index)];
        const auto out_index = get_reduced_offset(index, axis, keep_dims);
        auto &dest = output[offset(out_strides, out_index)];
        dest = reducer(dest, v);
    });
    apply(out_shape, [&](gsl::span<const size_t> index) -> void {
        auto &dest = output[offset(out_strides, index)];
        dest = post_process(dest);
    });
}
} // namespace

#define REDUCE_IMPL(op, reducer, post_process)                                 \
    case op:                                                                   \
        return reduce_impl(reducer, post_process,                              \
                           *reinterpret_cast<const T *>(init_value), input,    \
                           output, in_shape, axis, in_strides, out_shape,      \
                           out_strides, keep_dims)

#define REDUCE_IMPL_NO_POST(op, reducer)                                       \
    case op:                                                                   \
        return reduce_impl(reducer, identity<T>(),                             \
                           *reinterpret_cast<const T *>(init_value), input,    \
                           output, in_shape, axis, in_strides, out_shape,      \
                           out_strides, keep_dims)

template <typename T>
void reduce_prod(const T *input, T *output, gsl::span<const size_t> in_shape,
                 gsl::span<const size_t> in_strides,
                 gsl::span<const size_t> out_strides_origin,
                 gsl::span<const size_t> axes, bool keep_dims) noexcept {
    auto out_shape = get_reduced_shape(in_shape, axes, keep_dims);
    auto out_strides =
        out_strides_origin.size() == 0 ? dims_t{1} : dims_t(out_strides_origin);
    // init with init_value
    apply(gsl::make_span(out_shape).as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              output[offset(out_strides, index)] = 1;
          });

    apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        const auto src = input[offset(in_strides, index)];
        auto out_idx =
            offset(out_strides, get_reduced_offset(index, axes, keep_dims));
        auto &dst = output[out_idx];
        dst *= src;
    });
}

template <>
void reduce_prod(const bool *input, bool *output,
                 gsl::span<const size_t> in_shape,
                 gsl::span<const size_t> in_strides,
                 gsl::span<const size_t> out_strides_origin,
                 gsl::span<const size_t> axes, bool keep_dims) noexcept {
    auto out_shape = get_reduced_shape(in_shape, axes, keep_dims);
    auto out_strides =
        out_strides_origin.size() == 0 ? dims_t{1} : dims_t(out_strides_origin);
    // init with init_value
    apply(gsl::make_span(out_shape).as_span<const size_t>(),
          [&](gsl::span<const size_t> index) -> void {
              output[offset(out_strides, index)] = 1;
          });

    apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        const auto src = input[offset(in_strides, index)];
        auto out_idx =
            offset(out_strides, get_reduced_offset(index, axes, keep_dims));
        auto &dst = output[out_idx];
        dst &= src;
    });
}

template void reduce_prod<float>(const float *input, float *output,
                                 gsl::span<const size_t> in_shape,
                                 gsl::span<const size_t> in_strides,
                                 gsl::span<const size_t> out_strides,
                                 gsl::span<const size_t> axis,
                                 bool keep_dims) noexcept;

template <class T>
void reduce(reduce_op_t op, const T *init_value, const T *input, T *output,
            gsl::span<const size_t> in_shape, gsl::span<const size_t> axis,
            gsl::span<const size_t> in_strides,
            gsl::span<const size_t> out_strides, bool keep_dims) noexcept {
    auto out_shape = get_reduced_shape(in_shape, axis, keep_dims);
    switch (op) {
        REDUCE_IMPL(reduce_op_t::mean, [](T a, T b) { return a + b; },
                    [block_size = (T)get_reduce_block_size(in_shape, axis)](
                        T v) { return v / block_size; });
        REDUCE_IMPL_NO_POST(reduce_op_t::min,
                            [](T a, T b) { return a > b ? b : a; });
        REDUCE_IMPL_NO_POST(reduce_op_t::max,
                            [](T a, T b) { return a > b ? a : b; });
        REDUCE_IMPL_NO_POST(reduce_op_t::sum, [](T a, T b) { return a + b; });
        REDUCE_IMPL_NO_POST(reduce_op_t::sum_sqr,
                            [](T a, T b) { return a + (b * b); });
    case reduce_op_t::prod:
        return reduce_prod(reinterpret_cast<const T *>(input),
                           reinterpret_cast<T *>(output), in_shape, in_strides,
                           out_strides, axis, keep_dims);
    default:
        return;
    }
}

template <class T>
void reduce_sum_and_sum_sqr(const T *input, T *sum, T *sum_sqr,
                            gsl::span<const size_t> in_shape,
                            gsl::span<const size_t> in_strides,
                            gsl::span<const size_t> out_strides) {
    auto init_v = (T)0;
    kernels::reduce(reduce_op_t::sum, &init_v, input, sum, in_shape,
                    dims_t({in_shape.size() - 1}), in_strides, out_strides,
                    false);
    kernels::reduce(reduce_op_t::sum_sqr, &init_v, input, sum_sqr, in_shape,
                    dims_t({in_shape.size() - 1}), in_strides, out_strides,
                    false);
}

} // namespace kernels