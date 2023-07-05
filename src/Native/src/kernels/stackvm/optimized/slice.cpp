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
#include "opt_common.h"
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
template <size_t Dims, size_t CurDim = 0, class Callable = DefaultCallable>
void _slice_contiguous_dim_copy(const axes_t &begins,
                                NNCASE_UNUSED const axes_t &ends,
                                Callable &&line_copy, dims_t &in_index,
                                std::false_type) noexcept {
    in_index[Dims] = begins[Dims];
    line_copy();
}

template <size_t Dims, size_t CurDim = 0, class Callable = DefaultCallable>
void _slice_contiguous_dim_copy(const axes_t &begins,
                                NNCASE_UNUSED const axes_t &ends,
                                Callable &&line_copy, dims_t &in_index,
                                std::true_type) noexcept {
    for (size_t i = begins[CurDim]; i < static_cast<size_t>(ends[CurDim]);
         ++i) {
        in_index[CurDim] = i;
        _slice_contiguous_dim_copy<Dims, CurDim + 1>(
            begins, ends, std::forward<Callable>(line_copy), in_index,
            is_not_equal<Dims, CurDim + 1>);
    }
}

// optimized for n c h 1
size_t inline squeeze_dims(const gsl::span<const size_t> &in_shape) {
    return in_shape[in_shape.size() - 1] == 1 ? in_shape.size() - 1 - 1
                                              : in_shape.size() - 1;
}

template <class T>
result<void> slice_contiguous_impl(
    const T *input, T *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides,
    NNCASE_UNUSED gsl::span<const size_t> out_strides, const axes_t &begins,
    const axes_t &ends, NNCASE_UNUSED const axes_t &strides) noexcept {
    size_t elemsize = sizeof(T);
    auto *out_ptr = output;
    auto dims = squeeze_dims(in_shape);
    dims_t in_index(in_shape.size());

    auto line_copy = [&]() {
        const auto distance = static_cast<size_t>(ends[dims]) - begins[dims];
        const auto copy_size = distance * elemsize;
        const auto *in_ptr = input + offset(in_strides, in_index);
        opt_memcpy(out_ptr, in_ptr, copy_size);
        out_ptr += distance;
    };

    if (dims == 0) {
        _slice_contiguous_dim_copy<0>(begins, ends, line_copy, in_index,
                                      std::false_type{});
    } else if (dims == 1) {
        _slice_contiguous_dim_copy<1>(begins, ends, line_copy, in_index,
                                      std::true_type{});
    } else if (dims == 2) {
        _slice_contiguous_dim_copy<2>(begins, ends, line_copy, in_index,
                                      std::true_type{});
    } else if (dims == 3) {
        _slice_contiguous_dim_copy<3>(begins, ends, line_copy, in_index,
                                      std::true_type{});
    } else {
        assert(false);
    }
    return ok();
}

template <size_t Dims, size_t CurDim = 0, class Callable = DefaultCallable>
void _slice_dim_copy(NNCASE_UNUSED const axes_t &begins,
                     NNCASE_UNUSED const axes_t &ends,
                     NNCASE_UNUSED const axes_t &strides, Callable &&line_copy,
                     dims_t &in_index, dims_t &out_index,
                     std::false_type) noexcept {
    line_copy(in_index, out_index);
}

template <size_t Dims, size_t CurDim = 0, class Callable = DefaultCallable>
void _slice_dim_copy(const axes_t &begins, const axes_t &ends,
                     const axes_t &strides, Callable &&line_copy,
                     dims_t &in_index, dims_t &out_index,
                     std::true_type) noexcept {
    out_index[CurDim] = 0;
    for (size_t i = begins[CurDim]; i < static_cast<size_t>(ends[CurDim]);
         i += strides[CurDim]) {
        in_index[CurDim] = i;
        _slice_dim_copy<Dims, CurDim + 1>(
            begins, ends, strides, std::forward<Callable>(line_copy), in_index,
            out_index, is_not_equal<Dims, CurDim + 1>);
        ++out_index[CurDim];
    }
}

template <class Callable>
result<void> _slice_impl(gsl::span<const size_t> in_shape, const axes_t &begins,
                         const axes_t &ends, const axes_t &strides,
                         Callable &&line_copy) noexcept {
    auto dims = squeeze_dims(in_shape);
    dims_t in_index(in_shape.size());
    dims_t out_index(in_shape.size());
    if (dims == 0) {
        in_index[0] = begins[0];
        _slice_dim_copy<0>(begins, ends, strides,
                           std::forward<Callable &&>(line_copy), in_index,
                           out_index, std::false_type{});
    } else if (dims == 1) {
        _slice_dim_copy<1>(begins, ends, strides,
                           std::forward<Callable &&>(line_copy), in_index,
                           out_index, std::true_type{});
    } else if (dims == 2) {
        _slice_dim_copy<2>(begins, ends, strides,
                           std::forward<Callable &&>(line_copy), in_index,
                           out_index, std::true_type{});
    } else if (dims == 3) {
        _slice_dim_copy<3>(begins, ends, strides,
                           std::forward<Callable &&>(line_copy), in_index,
                           out_index, std::true_type{});
    } else {
        assert(false);
    }
    return ok();
}

template <class T>
result<void>
slice_linecopy_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                    gsl::span<const size_t> in_strides,
                    gsl::span<const size_t> out_strides, const axes_t &begins,
                    const axes_t &ends, const axes_t &strides) noexcept {
    auto dims = in_shape.size() - 1;
    return _slice_impl(in_shape, begins, ends, strides,
                       [&, dims](dims_t &in_index, dims_t &out_index) {
                           in_index[dims] = begins[dims];
                           const auto distance =
                               static_cast<size_t>(ends[dims]) - begins[dims];
                           auto copy_size = distance * sizeof(T);
                           const auto *in_ptr =
                               input + offset(in_strides, in_index);
                           auto *out_ptr =
                               output + offset(out_strides, out_index);
                           memcpy(out_ptr, in_ptr, copy_size);
                       });
}

template <class T>
result<void>
slice_strides_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                   gsl::span<const size_t> in_strides,
                   gsl::span<const size_t> out_strides, const axes_t &begins,
                   const axes_t &ends, const axes_t &strides) noexcept {
    auto dims = in_shape.size() - 1;
    return _slice_impl(in_shape, begins, ends, strides,
                       [&, dims](dims_t &in_index, dims_t &out_index) {
                           for (size_t i = begins[dims];
                                i < static_cast<size_t>(ends[dims]);
                                i += strides[dims]) {
                               in_index[dims] = i;
                               output[offset(out_strides, out_index)] =
                                   input[offset(in_strides, in_index)];
                               ++out_index[dims];
                           }
                           out_index[dims] = 0;
                       });
}
} // namespace

#define SLICE_LINECOPY_IMPL(size, type)                                        \
    case size:                                                                 \
        return slice_linecopy_impl(reinterpret_cast<const type *>(input),      \
                                   reinterpret_cast<type *>(output), in_shape, \
                                   in_strides, out_strides, begins, ends,      \
                                   strides)

#define SLICE_CONTIGUOUS_IMPL(size, type)                                      \
    case size:                                                                 \
        return slice_contiguous_impl(reinterpret_cast<const type *>(input),    \
                                     reinterpret_cast<type *>(output),         \
                                     in_shape, in_strides, out_strides,        \
                                     begins, ends, strides)

#define SLICE_STRIDES_IMPL(size, type)                                         \
    case size:                                                                 \
        return slice_strides_impl(reinterpret_cast<const type *>(input),       \
                                  reinterpret_cast<type *>(output), in_shape,  \
                                  in_strides, out_strides, begins, ends,       \
                                  strides)

result<void> nncase::kernels::stackvm::optimized::slice(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, const axes_t &begins,
    const axes_t &ends, const axes_t &strides,
    NNCASE_UNUSED kernel_context &context) noexcept {
    auto dims = begins.size();
    dims_t out_shape(dims);
    for (size_t i = 0; i < dims; ++i) {
        out_shape[i] = static_cast<size_t>(ends[i]) - begins[i];
    }

    for (size_t i = 0; i < dims; ++i) {
        if (strides[i] != 1) {
            // only last dims' stride is not 1
            if (strides[dims - 1] == 1) {
                TYPE_IMPL_SELECT(type, SLICE_LINECOPY_IMPL);
            } else {
                TYPE_IMPL_SELECT(type, SLICE_STRIDES_IMPL);
            }
        }
    }
    if (is_contiguous(in_shape, in_strides) &&
        is_contiguous(out_shape, out_strides)) {
        // all of strides are 1 and contiguous
        TYPE_IMPL_SELECT(type, SLICE_CONTIGUOUS_IMPL);
    } else {
        // summary memory is not continous, but line is contiguous
        TYPE_IMPL_SELECT(type, SLICE_LINECOPY_IMPL);
    }
}
