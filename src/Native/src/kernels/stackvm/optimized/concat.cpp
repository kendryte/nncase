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
#include "opt_ops.h"
#include <cstring>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

namespace {
template <size_t Axis, size_t CurAxis = 0, class Callable = DefaultCallable>
void _concat_contiguous_dim_copy(NNCASE_UNUSED gsl::span<const size_t> in_shape,
                                 NNCASE_UNUSED dims_t &in_index,
                                 Callable &&line_copy, std::false_type) {
    line_copy();
}

template <size_t Axis, size_t CurAxis = 0, class Callable = DefaultCallable>
void _concat_contiguous_dim_copy(gsl::span<const size_t> in_shape,
                                 dims_t &in_index, Callable &&line_copy,
                                 std::true_type) {
    for (size_t i = 0; i < in_shape[CurAxis]; ++i) {
        in_index[CurAxis] = i;
        _concat_contiguous_dim_copy<Axis, CurAxis + 1>(
            in_shape, in_index, std::forward<Callable>(line_copy),
            is_not_equal<Axis, CurAxis + 1>);
    }
}

template <class T>
result<void>
concat_contiguous_impl(gsl::span<const gsl::byte *const> inputs, T *output,
                       gsl::span<const size_t> out_shape,
                       gsl::span<const dims_t> &in_strides,
                       NNCASE_UNUSED gsl::span<const size_t> out_strides,
                       size_t axis, gsl::span<const size_t> concat_dims,
                       NNCASE_UNUSED kernel_context &context) noexcept {
    dims_t in_shape(out_shape), in_index(out_shape.size());
    auto subsize =
        std::accumulate(in_shape.begin() + (axis + 1), in_shape.end(), 1,
                        std::multiplies<size_t>());
    auto *out_ptr = output;
    auto line_copy = [&]() {
        for (size_t n = 0; n < inputs.size(); ++n) {
            const auto dims_width = concat_dims[n] * subsize;
            const auto in_offset = offset(in_strides[n], in_index);
            auto *in_ptr = reinterpret_cast<const T *>(inputs[n]) + in_offset;
            memcpy(out_ptr, in_ptr, dims_width * sizeof(T));
            out_ptr += dims_width;
        }
    };
    if (axis == 0) {
        _concat_contiguous_dim_copy<0>(in_shape, in_index, line_copy,
                                       std::false_type{});
    } else if (axis == 1) {
        _concat_contiguous_dim_copy<1>(in_shape, in_index, line_copy,
                                       std::true_type{});
    } else if (axis == 2) {
        _concat_contiguous_dim_copy<2>(in_shape, in_index, line_copy,
                                       std::true_type{});
    } else if (axis == 3) {
        _concat_contiguous_dim_copy<3>(in_shape, in_index, line_copy,
                                       std::true_type{});
    } else {
        assert(false);
    }
    return ok();
}

template <size_t N, size_t StartIndex = 0, class Callable = DefaultCallable>
void dim_n_for(NNCASE_UNUSED gsl::span<const size_t> in_shape,
               NNCASE_UNUSED dims_t &in_index, NNCASE_UNUSED dims_t &out_index,
               Callable &&dim_concat, std::false_type) {
    dim_concat(N);
}

// end, start
template <size_t N, size_t StartIndex = 0, class Callable = DefaultCallable>
void dim_n_for(gsl::span<const size_t> in_shape, dims_t &in_index,
               dims_t &out_index, Callable &&callable, std::true_type) {
    for (size_t channel = 0; channel < in_shape[StartIndex]; ++channel) {
        in_index[StartIndex] = channel;
        out_index[StartIndex] = channel;
        dim_n_for<N, StartIndex + 1>(in_shape, in_index, out_index,
                                     std::forward<Callable>(callable),
                                     is_not_equal<N, StartIndex + 1>);
    }
}

template <size_t Axis, class Callable>
void concat_inputs(gsl::span<const gsl::byte *const> inputs, dims_t &in_index,
                   dims_t &out_index, gsl::span<const size_t> concat_dims,
                   Callable &&copy_input_n) {
    out_index[Axis] = 0;
    for (size_t n = 0; n < inputs.size(); ++n) {
        for (size_t height = 0; height < concat_dims[n]; ++height) {
            in_index[Axis] = height;
            copy_input_n(n);
            ++out_index[Axis];
        }
    }
}

template <class T>
result<void> concat_impl(gsl::span<const gsl::byte *const> inputs, T *output,
                         gsl::span<const size_t> out_shape,
                         gsl::span<const dims_t> &in_strides,
                         gsl::span<const size_t> out_strides, size_t axis,
                         gsl::span<const size_t> concat_dims,
                         NNCASE_UNUSED kernel_context &context) noexcept {
    dims_t in_shape(out_shape);
    auto *out_ptr = output;
    auto dims = in_strides[0].size();
    dims_t out_index(dims);
    dims_t in_index(dims);
    auto line_copy = [&](size_t width, size_t n) {
        out_ptr = output + offset(out_strides, out_index);
        const auto *in_ptr = reinterpret_cast<const T *>(inputs[n]) +
                             offset(in_strides[n], in_index);
        memcpy(out_ptr, in_ptr, width * sizeof(T));
    };

    auto concat_last_dim = [&](size_t dim) {
        out_index[dim] = 0;
        for (size_t n = 0; n < inputs.size(); ++n) {
            const auto width = concat_dims[n];
            line_copy(width, n);
            // Dim3, axis 2
            out_index[dim] += width;
        }
    };

    if (dims == 1) {
        dim_n_for<0>(in_shape, in_index, out_index, concat_last_dim,
                     std::false_type{});
    } else if (dims == 2) {
        if (axis == 0) {
            const auto width = out_shape[1];
            concat_inputs<0>(inputs, in_index, out_index, concat_dims,
                             [&](size_t n) { line_copy(width, n); });
        } else {
            dim_n_for<1>(in_shape, in_index, out_index, concat_last_dim,
                         std::true_type{});
        }
    } else if (dims == 3) {
        if (axis == 0) {
            const auto width = out_shape[2];
            concat_inputs<0>(
                inputs, in_index, out_index, concat_dims, [&](size_t n) {
                    dim_n_for<2, 1>(
                        in_shape, in_index, out_index,
                        [&](NNCASE_UNUSED size_t dims) { line_copy(width, n); },
                        std::true_type{});
                });
        } else if (axis == 1) {
            const auto width = in_shape[2];
            dim_n_for<1>(
                in_shape, in_index, out_index,
                [&](NNCASE_UNUSED size_t dims) {
                    concat_inputs<1>(inputs, in_index, out_index, concat_dims,
                                     [&](size_t n) { line_copy(width, n); });
                },
                std::true_type{});
        } else {
            dim_n_for<2>(in_shape, in_index, out_index, concat_last_dim,
                         std::true_type{});
        }
    } else if (dims == 4) {
        if (axis == 0) {
            const auto width = out_shape[3];
            concat_inputs<0>(
                inputs, in_index, out_index, concat_dims, [&](size_t n) {
                    dim_n_for<3, 1>(
                        in_shape, in_index, out_index,
                        [&](NNCASE_UNUSED size_t dims) { line_copy(width, n); },
                        std::true_type{});
                });
        } else if (axis == 1) {
            const auto width = out_shape[3];
            dim_n_for<1>(
                in_shape, in_index, out_index,
                [&](NNCASE_UNUSED size_t dims) {
                    concat_inputs<1>(inputs, in_index, out_index, concat_dims,
                                     [&](size_t n) {
                                         dim_n_for<3, 2>(
                                             in_shape, in_index, out_index,
                                             [&](NNCASE_UNUSED size_t dims) {
                                                 line_copy(width, n);
                                             },
                                             std::true_type{});
                                     });
                },
                std::true_type{});
        } else if (axis == 2) {
            const auto width = out_shape[3];
            dim_n_for<2>(
                in_shape, in_index, out_index,
                [&](NNCASE_UNUSED size_t dims) {
                    concat_inputs<2>(inputs, in_index, out_index, concat_dims,
                                     [&](size_t n) { line_copy(width, n); });
                },
                std::true_type{});
        } else {
            dim_n_for<3>(in_shape, in_index, out_index, concat_last_dim,
                         std::true_type{});
        }
    }
    return ok();
}
} // namespace

#define CONCAT_IMPL(size, type)                                                \
    case size:                                                                 \
        return concat_impl(inputs, reinterpret_cast<type *>(output),           \
                           out_shape, in_strides, out_strides, axis,           \
                           concat_dims, context)

#define CONCAT_CONTIGUOUS_IMPL(size, type)                                     \
    case size:                                                                 \
        return concat_contiguous_impl(                                         \
            inputs, reinterpret_cast<type *>(output), out_shape, in_strides,   \
            out_strides, axis, concat_dims, context)

result<void> optimized::concat(datatype_t type,
                               gsl::span<const gsl::byte *const> inputs,
                               gsl::byte *output,
                               gsl::span<const size_t> out_shape,
                               gsl::span<const dims_t> in_strides,
                               gsl::span<const size_t> out_strides, size_t axis,
                               gsl::span<const size_t> concat_dims,
                               kernel_context &context) noexcept {
    dims_t in_shape(out_shape);
    if (!is_contiguous(out_shape, out_strides)) {
        TYPE_IMPL_SELECT(type, CONCAT_IMPL);
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto tmp = in_shape[axis];
        in_shape[axis] = concat_dims[i];
        if (!is_contiguous(in_shape, in_strides[i])) {
            TYPE_IMPL_SELECT(type, CONCAT_IMPL);
        }
        in_shape[axis] = tmp;
    }
    TYPE_IMPL_SELECT(type, CONCAT_CONTIGUOUS_IMPL);
}
