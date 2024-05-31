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
#include <iostream>

template <typename T>
result<void> lstm_impl(const T *input, const T *w_xc, const T *w_rc,
                       [[maybe_unused]] const T *bias, const T *init_h,
                       const T *init_c, T *output, T *output_h, T *output_c,
                       std::span<const size_t> in_shape_3,
                       std::span<const size_t> init_h_shape_3,
                       std::span<const size_t> init_c_shape_3,
                       std::span<const size_t> out_shape_3,
                       std::span<const size_t> w_xc_shape_3,
                       std::span<const size_t> w_rc_shape_3,
                       lstmdirection_t direction) {
    auto in_shape = to_4d(in_shape_3);
    auto init_h_shape = to_4d(init_h_shape_3);
    auto init_c_shape = to_4d(init_c_shape_3);
    auto w_xc_shape = to_4d(w_xc_shape_3);
    auto w_rc_shape = to_4d(w_rc_shape_3);
    auto out_shape = to_4d(out_shape_3);

    auto tanh = [&](T x) {
        return (1 - exp(-2 * (float)x)) / (1 + exp(-2 * (float)x));
    };
    auto sigmoid = [&](T x) { return 1 / (1 + exp(-x)); };

    auto output_h_tmp = std::make_unique<T[]>(compute_size(init_h_shape));
    auto output_c_tmp = std::make_unique<T[]>(compute_size(init_c_shape));
    std::memcpy(output_h_tmp.get(), init_h,
                sizeof(T) * compute_size(init_h_shape));
    std::memcpy(output_c_tmp.get(), init_c,
                sizeof(T) * compute_size(init_c_shape));

    auto hidden_size = w_xc_shape[2];
    std::vector<uint32_t> seq_len_loop;
    for (uint32_t l = 0; l < in_shape[1]; l++)
        seq_len_loop.push_back(l);
    if (direction == lstmdirection_t::reverse)
        std::reverse(seq_len_loop.begin(), seq_len_loop.end());
    // d: num_directions
    for (uint32_t d = 0; d < out_shape[1]; d++) {
        if (d == 1)
            std::reverse(seq_len_loop.begin(), seq_len_loop.end());
        for (uint32_t b = 0; b < in_shape[2]; b++) {
            for (auto &l : seq_len_loop) {
                // g = w_xc_x + w_xc_x
                auto out_mul1 = std::vector<T>(out_shape[3] * 4);
                auto out_mul2 = std::vector<T>(out_shape[3] * 4);
                for (size_t o = 0; o < out_mul1.size(); o++) {
                    for (size_t i = 0; i < in_shape[3]; i++) {
                        auto in_idx =
                            i + b * in_shape[3] + l * in_shape[2] * in_shape[3];
                        auto w_idx = i + o * w_xc_shape[3] +
                                     d * w_xc_shape[2] * w_xc_shape[3];

                        out_mul1[o] += T(input[in_idx]) * T(w_xc[w_idx]);
                    }
                    auto b_idx1 = d * w_rc_shape[2] + o;
                    out_mul1[o] += bias[b_idx1];

                    for (size_t i = 0; i < out_shape[3]; i++) {
                        auto in_idx = i + b * out_shape[3] +
                                      d * out_shape[2] * out_shape[3];
                        auto w_idx = i + o * w_rc_shape[3] +
                                     d * w_rc_shape[2] * w_rc_shape[3];
                        out_mul2[o] += T(output_h_tmp[in_idx]) * T(w_rc[w_idx]);
                    }
                    auto b_idx2 = d * w_rc_shape[2] + hidden_size + o;
                    out_mul2[o] += bias[b_idx2];

                    out_mul1[o] += out_mul2[o];
                }

                // ft = sigmoid(g[2])
                for (size_t o = 0; o < out_shape[3]; o++) {
                    out_mul1[o + out_shape[3] * 2] =
                        sigmoid(out_mul1[o + out_shape[3] * 2]);
                }

                // ct = init_c * ft
                for (size_t o = 0; o < out_shape[3]; o++) {
                    out_mul1[o + out_shape[3] * 2] =
                        out_mul1[o + out_shape[3] * 2] *
                        output_c_tmp[o + b * out_shape[3] +
                                     d * out_shape[2] * out_shape[3]];
                }

                // it = sigmoid(g[0])
                for (size_t o = 0; o < out_shape[3]; o++) {
                    out_mul1[o + out_shape[3] * 0] =
                        sigmoid(out_mul1[o + out_shape[3] * 0]);
                }

                // c_t = tanh(g[3])
                for (size_t o = 0; o < out_shape[3]; o++) {
                    out_mul1[o + out_shape[3] * 3] =
                        tanh(out_mul1[o + out_shape[3] * 3]);
                }

                // c_t_it = it * c_t
                for (size_t o = 0; o < out_shape[3]; o++) {
                    out_mul1[o + out_shape[3] * 0] =
                        out_mul1[o + out_shape[3] * 0] *
                        out_mul1[o + out_shape[3] * 3];
                }

                // ct = ct + c_t_it
                for (size_t o = 0; o < out_shape[3]; o++) {
                    output_c_tmp[o + d * out_shape[2] * out_shape[3]] =
                        T(out_mul1[o + out_shape[3] * 2] +
                          out_mul1[o + out_shape[3] * 0]);
                }

                // ot = sigmoid(g[1])
                for (size_t o = 0; o < out_shape[3]; o++) {
                    out_mul1[o + out_shape[3] * 1] =
                        sigmoid(out_mul1[o + out_shape[3] * 1]);
                }

                // tanh_ct = tanh(ct_o)
                for (size_t o = 0; o < out_shape[3]; o++) {
                    out_mul1[o + out_shape[3] * 3] = tanh(float(
                        output_c_tmp[o + d * out_shape[2] * out_shape[3]]));
                }

                // ht = ot * tanh_ct
                for (size_t o = 0; o < out_shape[3]; o++) {
                    output_h_tmp[o + d * out_shape[2] * out_shape[3]] =
                        T(out_mul1[o + out_shape[3] * 3] *
                          out_mul1[o + out_shape[3] * 1]);
                }
                std::memcpy(output + b * out_shape[3] +
                                d * out_shape[2] * out_shape[3] +
                                l * out_shape[1] * out_shape[2] * out_shape[3],
                            output_h_tmp.get() +
                                d * out_shape[2] * out_shape[3],
                            sizeof(T) * out_shape[3]);

                if (l == seq_len_loop.back()) {
                    std::memcpy(output_h + b * out_shape[3] +
                                    d * out_shape[2] * out_shape[3],
                                output_h_tmp.get() +
                                    d * out_shape[2] * out_shape[3],
                                sizeof(T) * out_shape[3]);
                    std::memcpy(output_c + b * out_shape[3] +
                                    d * out_shape[2] * out_shape[3],
                                output_c_tmp.get() +
                                    d * out_shape[2] * out_shape[3],
                                sizeof(T) * out_shape[3]);
                }
            }
        }
    }
    return ok();
}

#define LSTM_IMPL(type)                                                        \
    return lstm_impl(                                                          \
        IN_CAST(type, input), IN_CAST(type, w_xc), IN_CAST(type, w_rc),        \
        IN_CAST(type, bias), IN_CAST(type, init_h), IN_CAST(type, init_c),     \
        OUT_CAST(type, output), OUT_CAST(type, output_h),                      \
        OUT_CAST(type, output_c), in_shape_3, init_h_shape_3, init_c_shape_3,  \
        out_shape_3, w_xc_shape_3, w_rc_shape_3, direction);

#define TYPE_SELECT_LSTM(_typecode, _impl)                                     \
    switch (_typecode) {                                                       \
    case dt_float32:                                                           \
        _impl(float);                                                          \
    case dt_float16:                                                           \
        _impl(half);                                                           \
    case dt_float64:                                                           \
        _impl(double);                                                         \
    default:                                                                   \
        return err(std::errc::not_supported);                                  \
    }

result<void> nncase::kernels::stackvm::reference::lstm(
    typecode_t type, const std::byte *input, const std::byte *w_xc,
    const std::byte *w_rc, [[maybe_unused]] const std::byte *bias,
    const std::byte *init_h, const std::byte *init_c, std::byte *output,
    std::byte *output_h, std::byte *output_c,
    std::span<const size_t> in_shape_3, std::span<const size_t> init_h_shape_3,
    std::span<const size_t> init_c_shape_3, std::span<const size_t> out_shape_3,
    std::span<const size_t> w_xc_shape_3, std::span<const size_t> w_rc_shape_3,
    lstmdirection_t direction) {
    TYPE_SELECT_LSTM(type, LSTM_IMPL);
}
