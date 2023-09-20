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
                       gsl::span<const size_t> in_shape_3,
                       gsl::span<const size_t> init_h_shape_3,
                       gsl::span<const size_t> init_c_shape_3,
                       gsl::span<const size_t> out_shape_3,
                       gsl::span<const size_t> w_xc_shape_3,
                       gsl::span<const size_t> w_rc_shape_3,
                       lstmdirection_t direction) {
    auto in_shape = to_4d(in_shape_3);
    auto init_h_shape = to_4d(init_h_shape_3);
    auto init_c_shape = to_4d(init_c_shape_3);
    auto w_xc_shape = to_4d(w_xc_shape_3);
    auto w_rc_shape = to_4d(w_rc_shape_3);
    auto out_shape = to_4d(out_shape_3);

    auto tanh = [&](float x) { return (1 - exp(-2 * x)) / (1 + exp(-2 * x)); };
    auto sigmoid = [&](float x) { return 1 / (1 + exp(-x)); };

    auto output_h_tmp = std::make_unique<float[]>(compute_size(init_h_shape));
    auto output_c_tmp = std::make_unique<float[]>(compute_size(init_c_shape));
    std::memcpy(output_h_tmp.get(), init_h,
                sizeof(float) * compute_size(init_h_shape));
    std::memcpy(output_c_tmp.get(), init_c,
                sizeof(float) * compute_size(init_c_shape));

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
                auto out_mul1 = std::vector<float>(out_shape[3] * 4);
                auto out_mul2 = std::vector<float>(out_shape[3] * 4);
                for (size_t o = 0; o < out_mul1.size(); o++) {
                    for (size_t i = 0; i < in_shape[3]; i++) {
                        auto in_idx =
                            i + b * in_shape[3] + l * in_shape[2] * in_shape[3];
                        auto w_idx = i + o * w_xc_shape[3] +
                                     d * w_xc_shape[2] * w_xc_shape[3];

                        out_mul1[o] +=
                            float(input[in_idx]) * float(w_xc[w_idx]);
                    }
                    auto b_idx1 = d * w_rc_shape[2] + o;
                    out_mul1[o] += bias[b_idx1];

                    for (size_t i = 0; i < out_shape[3]; i++) {
                        auto in_idx = i + b * out_shape[3] +
                                      d * out_shape[2] * out_shape[3];
                        auto w_idx = i + o * w_rc_shape[3] +
                                     d * w_rc_shape[2] * w_rc_shape[3];
                        out_mul2[o] +=
                            float(output_h_tmp[in_idx]) * float(w_rc[w_idx]);
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
                        float(out_mul1[o + out_shape[3] * 2] +
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
                        float(out_mul1[o + out_shape[3] * 3] *
                              out_mul1[o + out_shape[3] * 1]);
                }
                std::memcpy(output + b * out_shape[3] +
                                d * out_shape[2] * out_shape[3] +
                                l * out_shape[1] * out_shape[2] * out_shape[3],
                            output_h_tmp.get() +
                                d * out_shape[2] * out_shape[3],
                            sizeof(float) * out_shape[3]);

                if (l == seq_len_loop.back()) {
                    std::memcpy(output_h + b * out_shape[3] +
                                    d * out_shape[2] * out_shape[3],
                                output_h_tmp.get() +
                                    d * out_shape[2] * out_shape[3],
                                sizeof(float) * out_shape[3]);
                    std::memcpy(output_c + b * out_shape[3] +
                                    d * out_shape[2] * out_shape[3],
                                output_c_tmp.get() +
                                    d * out_shape[2] * out_shape[3],
                                sizeof(float) * out_shape[3]);
                }
            }
        }
    }
    return ok();
}

result<void> nncase::kernels::stackvm::reference::lstm(
    const float *input, const float *w_xc, const float *w_rc,
    [[maybe_unused]] const float *bias, const float *init_h,
    const float *init_c, float *output, float *output_h, float *output_c,
    gsl::span<const size_t> in_shape_3, gsl::span<const size_t> init_h_shape_3,
    gsl::span<const size_t> init_c_shape_3, gsl::span<const size_t> out_shape_3,
    gsl::span<const size_t> w_xc_shape_3, gsl::span<const size_t> w_rc_shape_3,
    lstmdirection_t direction) {
    return lstm_impl(input, w_xc, w_rc, bias, init_h, init_c, output, output_h,
                     output_c, in_shape_3, init_h_shape_3, init_c_shape_3,
                     out_shape_3, w_xc_shape_3, w_rc_shape_3, direction);
}
