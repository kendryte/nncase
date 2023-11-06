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
#include <iostream>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template result<void>
reference::gru<float>(const float *input, const float *w, const float *r, const float *b, float *initial_h,
    float *output, float *output_h, const runtime_shape_t &input_shape,
    const runtime_shape_t &w_shape, int mode, bool linear_before_reset) noexcept;

template <typename T>
result<void> reference::gru(const T *input, const T *w, const T *r, const T *b, T *initial_h, T *output, T *output_h,
    const runtime_shape_t &input_shape, const runtime_shape_t &w_shape, int mode,
    bool linear_before_reset) noexcept
{
    const int seq_length = input_shape[0];
    const int batch_size = input_shape[1];
    const int input_size = input_shape[2];
    const int num_direction = w_shape[0];
    const int hidden_size = w_shape[1] / 3;

    auto sigmoid = [&](float x) {
        return 1 / (1 + std::exp(-x));
    };
    auto tanh = [&](float x) {
        return std::tanh(x);
    };
    // copy input to output
    runtime_shape_t out_shape { (size_t)seq_length, (size_t)num_direction, (size_t)batch_size, (size_t)hidden_size };

    auto x_gate_size = batch_size * input_size;
    auto w_gate_size = 3 * hidden_size * input_size;
    auto h_t_size = batch_size * hidden_size;
    auto r_gate_size = 3 * hidden_size * hidden_size;

    auto tmp_a = std::vector<float>(batch_size * hidden_size, 0.f);
    auto tmp_b = std::vector<float>(batch_size * hidden_size, 0.f);
    auto gate_z = std::vector<float>(batch_size * hidden_size, 0.f);
    auto gate_r = std::vector<float>(batch_size * hidden_size, 0.f);
    auto gate_h = std::vector<float>(batch_size * hidden_size, 0.f);

    std::vector<int> seq_len_loop;
    for (int l = 0; l < seq_length; l++)
        seq_len_loop.push_back(l);
    if (mode == lstm_direction::kReverse)
        std::reverse(seq_len_loop.begin(), seq_len_loop.end());
    auto x_i = input;
    auto h_t = initial_h;
    auto w_i = w;
    auto r_i = r;
    auto b_i = b;
    for (int d = 0; d < num_direction; d++)
    {
        h_t = initial_h + d * h_t_size;
        w_i = w + d * w_gate_size;
        r_i = r + d * r_gate_size;
        b_i = b + d * 6 * hidden_size;
        if (d == 1)
            std::reverse(seq_len_loop.begin(), seq_len_loop.end());
        for (auto i : seq_len_loop)
        {
            x_i = input + i * x_gate_size;
            // clean gate_z gate_r gate_h
            std::fill(gate_z.begin(), gate_z.end(), 0.f);
            std::fill(gate_r.begin(), gate_r.end(), 0.f);
            std::fill(gate_h.begin(), gate_h.end(), 0.f);

            // clean tmp_a tmp_b
            std::fill(tmp_a.begin(), tmp_a.end(), 0.f);
            std::fill(tmp_b.begin(), tmp_b.end(), 0.f);
            // gate_z = x_i * w_i_z + b_w_z + h_t *r_i_z + b_r_z
            for (int bs = 0; bs < batch_size; bs++)
            {
                for (int hs = 0; hs < hidden_size; hs++)
                {
                    for (int is = 0; is < input_size; is++)
                    {
                        tmp_a[bs * hidden_size + hs] += x_i[bs * input_size + is] * w_i[hs * input_size + is];
                    }
                    tmp_a[bs * hidden_size + hs] += b_i[hs];
                    for (int rs = 0; rs < hidden_size; rs++)
                    {
                        tmp_b[bs * hidden_size + hs] += h_t[bs * hidden_size + rs] * r_i[hs * hidden_size + rs];
                    }
                    tmp_b[bs * hidden_size + hs] += b_i[3 * hidden_size + hs];
                    gate_z[bs * hidden_size + hs] = tmp_a[bs * hidden_size + hs] + tmp_b[bs * hidden_size + hs];
                }
            }
            // gate_z = sigmoid(gate_z);
            std::transform(gate_z.begin(), gate_z.end(), gate_z.begin(), sigmoid);

            // clear tmp_a tmp_b
            std::fill(tmp_a.begin(), tmp_a.end(), 0.f);
            std::fill(tmp_b.begin(), tmp_b.end(), 0.f);
            // gate_r = x_i * w_i_r + b_w_r + h_t *r_i_r + b_r_r
            for (int bs = 0; bs < batch_size; bs++)
            {
                for (int hs = 0; hs < hidden_size; hs++)
                {
                    for (int is = 0; is < input_size; is++)
                    {
                        tmp_a[bs * hidden_size + hs] += x_i[bs * input_size + is] * w_i[hidden_size * input_size + hs * input_size + is];
                    }
                    tmp_a[bs * hidden_size + hs] += b_i[hidden_size + hs];
                    for (int rs = 0; rs < hidden_size; rs++)
                    {
                        tmp_b[bs * hidden_size + hs] += h_t[bs * hidden_size + rs] * r_i[hidden_size * hidden_size + hs * hidden_size + rs];
                    }
                    tmp_b[bs * hidden_size + hs] += b_i[4 * hidden_size + hs];
                    gate_r[bs * hidden_size + hs] = tmp_a[bs * hidden_size + hs] + tmp_b[bs * hidden_size + hs];
                }
            }
            // gate_r = sigmoid(gate_r);
            std::transform(gate_r.begin(), gate_r.end(), gate_r.begin(), sigmoid);

            // clear tmp_a tmp_b
            std::fill(tmp_a.begin(), tmp_a.end(), 0.f);
            std::fill(tmp_b.begin(), tmp_b.end(), 0.f);
            // gate_h = x_i * w_i_h + b_w_h + gate_r·h_t *r_i_h + b_r_h
            for (int bs = 0; bs < batch_size; bs++)
            {
                for (int hs = 0; hs < hidden_size; hs++)
                {
                    for (int is = 0; is < input_size; is++)
                    {
                        tmp_a[bs * hidden_size + hs] += x_i[bs * input_size + is] * w_i[2 * hidden_size * input_size + hs * input_size + is];
                    }
                    tmp_a[bs * hidden_size + hs] += b_i[2 * hidden_size + hs];

                    for (int rs = 0; rs < hidden_size; rs++)
                    {
                        if (!linear_before_reset)
                            tmp_b[bs * hidden_size + hs] += gate_r[bs * hidden_size + rs] * h_t[bs * hidden_size + rs] * r_i[2 * hidden_size * hidden_size + hs * hidden_size + rs];
                        else
                            tmp_b[bs * hidden_size + hs] += h_t[bs * hidden_size + rs] * r_i[2 * hidden_size * hidden_size + hs * hidden_size + rs];
                    }
                    tmp_b[bs * hidden_size + hs] += b_i[5 * hidden_size + hs];

                    if (!linear_before_reset)
                        gate_h[bs * hidden_size + hs] = tmp_a[bs * hidden_size + hs] + tmp_b[bs * hidden_size + hs];
                    else
                        gate_h[bs * hidden_size + hs] = tmp_a[bs * hidden_size + hs] + gate_r[bs * hidden_size + hs] * tmp_b[bs * hidden_size + hs];
                }
            }
            // gate_h = tanh(gate_h);
            std::transform(gate_h.begin(), gate_h.end(), gate_h.begin(), tanh);

            for (int k = 0; k < batch_size * hidden_size; k++)
            {
                h_t[k] = (1 - gate_z[k]) * gate_h[k] + gate_z[k] * h_t[k];
                // *output++ = h_t[k];
                output[i * (num_direction * batch_size * hidden_size) + d * (batch_size * hidden_size) + k] = h_t[k];
            }
        }
        // if (mode == lstm_direction::kReverse || d == 1)
        //     h_t.reverse();
        for (int k = 0; k < batch_size * hidden_size; k++)
        {
            output_h[d * (batch_size * hidden_size) + k] = h_t[k];
        }
    }

    return ok();
}
