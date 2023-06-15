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
#include "ref_ops.h"
#include <cstring>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
dims_t get_padded_shape(gsl::span<const size_t> in_shape,
                        const paddings_t &paddings) {
    dims_t out_shape(in_shape.size());
    for (size_t i = 0; i < in_shape.size(); i++)
        out_shape[i] = (size_t)((int32_t)in_shape[i] + paddings[i].sum() +
                                (in_shape[i] - 1) * paddings[i].interior);
    return out_shape;
}

dims_t get_in_index(gsl::span<const size_t> index,
                    gsl::span<const size_t> in_shape,
                    const paddings_t &paddings, pad_mode_t mode,
                    bool &pad_element) {
    dims_t in_index(index.size());
    pad_element = false;
    for (size_t i = 0; i < index.size(); i++) {
        auto &padding = paddings[i];
        if ((int32_t)index[i] < padding.before) {
            pad_element = true;
            if (mode == pad_mode_t::reflect)
                in_index[i] = (size_t)padding.before - index[i];
            else if (mode == pad_mode_t::symmetric)
                in_index[i] = (size_t)padding.before - index[i] - 1;
            else if (mode == pad_mode_t::edge)
                in_index[i] = 0;
        } else {
            auto cnt_idx = (int32_t)index[i] - padding.before;
            if (cnt_idx > (int32_t)in_shape[i] - 1) {
                pad_element = true;
                if (mode == pad_mode_t::reflect)
                    in_index[i] =
                        in_shape[i] - 2 - ((size_t)cnt_idx - in_shape[i]);
                else if (mode == pad_mode_t::symmetric)
                    in_index[i] =
                        in_shape[i] - 1 - ((size_t)cnt_idx - in_shape[i]);
                else if (mode == pad_mode_t::edge)
                    in_index[i] = in_shape[i] - 1;
            } else {
                in_index[i] = (size_t)cnt_idx;
            }
        }
    }

    return in_index;
}

template <class T>
result<void>
pad_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
         gsl::span<const size_t> out_shape, gsl::span<const size_t> in_strides,
         gsl::span<const size_t> out_strides, const paddings_t &paddings,
         pad_mode_t mode, T pad_value,
         NNCASE_UNUSED kernel_context &context) noexcept {
    int sum = 0;
    for (const auto &item : paddings) {
        sum += item.sum();
    }
    if (sum == 0) {
        auto out_size = compute_size(out_shape);
        memcpy(output, input, out_size);
        return ok();
    }

    return apply(out_shape, [&](gsl::span<const size_t> index) -> result<void> {
        bool pad_element = false;
        auto in_index =
            get_in_index(index, in_shape, paddings, mode, pad_element);
        T value;
        if (!pad_element || mode != pad_mode_t::constant)
            value = input[offset(in_strides, in_index)];
        else
            value = pad_value;
        output[offset(out_strides, index)] = value;
        return ok();
    });
}

template <class T>
result<void> interior_pad_impl(const T *input, T *output,
                               NNCASE_UNUSED gsl::span<const size_t> in_shape,
                               gsl::span<const size_t> out_shape,
                               NNCASE_UNUSED gsl::span<const size_t> in_strides,
                               gsl::span<const size_t> out_strides,
                               const paddings_t &paddings, T pad_value,
                               NNCASE_UNUSED kernel_context &context) noexcept {
    size_t idx = 0;
    size_t size = paddings.size();
    assert(size >= 2);
    size_t h_axis = size - 2;
    size_t w_axis = size - 1;
    bool h_pad = paddings[h_axis].interior != 0 ? true : false;
    bool w_pad = paddings[w_axis].interior != 0 ? true : false;

    return apply(out_shape, [&](gsl::span<const size_t> index) -> result<void> {
        bool pad_element =
            (h_pad && (index[h_axis] % (paddings[h_axis].interior + 1) != 0)) ||
            (w_pad && (index[w_axis] % (paddings[w_axis].interior + 1) != 0));
        T value = pad_element ? pad_value : input[idx++];
        assert(idx <= compute_size(in_shape));
        output[offset(out_strides, index)] = value;
        return ok();
    });
}

} // namespace

result<void> nncase::kernels::stackvm::reference::pad(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, const paddings_t &paddings,
    pad_mode_t mode, const gsl::byte *pad_value,
    kernel_context &context) noexcept {
    auto unit = runtime::get_bytes(type);
    if (std::all_of(paddings.begin(), paddings.end(),
                    [](const padding &p) { return p.interior == 0; })) {
        auto out_shape = get_padded_shape(in_shape, paddings);
        switch (unit) {
        case 1:
            return pad_impl(reinterpret_cast<const uint8_t *>(input),
                            reinterpret_cast<uint8_t *>(output), in_shape,
                            out_shape, in_strides, out_strides, paddings, mode,
                            *IN_CAST(uint8_t, pad_value), context);

        case 2:
            return pad_impl(reinterpret_cast<const uint16_t *>(input),
                            reinterpret_cast<uint16_t *>(output), in_shape,
                            out_shape, in_strides, out_strides, paddings, mode,
                            *IN_CAST(uint16_t, pad_value), context);

        case 4:
            return pad_impl(reinterpret_cast<const uint32_t *>(input),
                            reinterpret_cast<uint32_t *>(output), in_shape,
                            out_shape, in_strides, out_strides, paddings, mode,
                            *IN_CAST(uint32_t, pad_value), context);
        case 8:
            return pad_impl(reinterpret_cast<const uint64_t *>(input),
                            reinterpret_cast<uint64_t *>(output), in_shape,
                            out_shape, in_strides, out_strides, paddings, mode,
                            *IN_CAST(uint64_t, pad_value), context);
        default:
            return err(std::errc::not_supported);
        }
    } else {
        assert(mode == pad_mode_t::constant);

        // interior padding
        paddings_t padding_cfg = paddings;
        for (auto &p : padding_cfg) {
            p.before = 0;
            p.after = 0;
        }

        auto out_shape = get_padded_shape(in_shape, padding_cfg);
        auto out_size = compute_size(out_shape);
        auto strides = get_default_strides(out_shape);
        std::vector<uint8_t> v(out_size * unit, 0);

        switch (unit) {
        case 1: {
            NNCASE_UNUSED auto ret = interior_pad_impl(
                reinterpret_cast<const uint8_t *>(input),
                reinterpret_cast<uint8_t *>(v.data()), in_shape, out_shape,
                in_strides, strides, padding_cfg, *IN_CAST(uint8_t, pad_value),
                context);
            break;
        }

        case 2: {
            NNCASE_UNUSED auto ret = interior_pad_impl(
                reinterpret_cast<const uint16_t *>(input),
                reinterpret_cast<uint16_t *>(v.data()), in_shape, out_shape,
                in_strides, strides, padding_cfg, *IN_CAST(uint16_t, pad_value),
                context);
            break;
        }
        case 4: {
            NNCASE_UNUSED auto ret = interior_pad_impl(
                reinterpret_cast<const uint32_t *>(input),
                reinterpret_cast<uint32_t *>(v.data()), in_shape, out_shape,
                in_strides, strides, padding_cfg, *IN_CAST(uint32_t, pad_value),
                context);
            break;
        }
        default:
            return err(std::errc::not_supported);
        }

        // edge padding
        padding_cfg = paddings;
        for (auto &p : padding_cfg) {
            p.interior = 0;
        }

        auto out_shape2 = get_padded_shape(out_shape, padding_cfg);
        switch (unit) {
        case 1:
            return pad_impl(reinterpret_cast<const uint8_t *>(v.data()),
                            reinterpret_cast<uint8_t *>(output), out_shape,
                            out_shape2, strides, out_strides, padding_cfg, mode,
                            *IN_CAST(uint8_t, pad_value), context);

        case 2:
            return pad_impl(reinterpret_cast<const uint16_t *>(v.data()),
                            reinterpret_cast<uint16_t *>(output), out_shape,
                            out_shape2, strides, out_strides, padding_cfg, mode,
                            *IN_CAST(uint16_t, pad_value), context);

        case 4:
            return pad_impl(reinterpret_cast<const uint32_t *>(v.data()),
                            reinterpret_cast<uint32_t *>(output), out_shape,
                            out_shape2, strides, out_strides, padding_cfg, mode,
                            *IN_CAST(uint32_t, pad_value), context);

        case 8:
            return pad_impl(reinterpret_cast<const uint64_t *>(v.data()),
                            reinterpret_cast<uint64_t *>(output), out_shape,
                            out_shape2, strides, out_strides, padding_cfg, mode,
                            *IN_CAST(uint64_t, pad_value), context);
        default:
            return err(std::errc::not_supported);
        }
    }
}