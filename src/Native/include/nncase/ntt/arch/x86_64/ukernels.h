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
#include "../../ukernels.h"
#include "arch_types.h"
#include "nncase/ntt/vector.h"
#include <vector>

namespace nncase::ntt::ukernels {
template <size_t M, size_t N, size_t MStrides>
class u_pack<M, N, MStrides, true, float, vector<float, 8>> {
  public:
    constexpr void operator()(const float *input,
                              vector<float, 8> *output) noexcept {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < M; i++) {
                output[j](i) = input[i * MStrides + j];
            }
        }

        if constexpr (M < 8) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = M; i < 8; i++) {
                    output[j](i) = 0.f;
                }
            }
        }
    }
};

template <reduce_op Op> struct u_reduce<Op, vector<float, 8>, true> {
  public:
    constexpr vector<float, 8>
    operator()(const vector<float, 8> *input, size_t input_stride, size_t count,
               vector<float, 8> init_value) noexcept {
        using binary_op_t =
            typename reduce_to_binary_type<Op>::template type<vector<float, 8>,
                                                              vector<float, 8>>;
        binary_op_t op;
        if (count / 4) {
            vector<float, 8> tmp[4];
            for (size_t i = 0; i < 4; i++) {
                tmp[i] = input[i * input_stride];
            }
            input += input_stride * 4;
            count -= 4;
            while (count / 4) {
                for (size_t i = 0; i < 4; i++) {
                    tmp[i] = op(tmp[i], input[i * input_stride]);
                }
                input += input_stride * 4;
                count -= 4;
            }

            tmp[0] = op(tmp[0], tmp[1]);
            tmp[2] = op(tmp[2], tmp[3]);
            tmp[0] = op(tmp[0], tmp[2]);
            init_value = op(init_value, tmp[0]);
        }

        if (count / 2) {
            vector<float, 8> tmp[2];
            for (size_t i = 0; i < 2; i++) {
                tmp[i] = input[i * input_stride];
            }
            input += input_stride * 2;
            count -= 2;
            while (count / 2) {
                for (size_t i = 0; i < 2; i++) {
                    tmp[i] = op(tmp[i], input[i * input_stride]);
                }
                input += input_stride * 2;
                count -= 2;
            }

            tmp[0] = op(tmp[0], tmp[1]);
            init_value = op(init_value, tmp[0]);
        }

        for (size_t i = 0; i < count; i++) {
            init_value = op(init_value, *input);
            input += input_stride;
        }
        return init_value;
    }
};

template <reduce_op Op> struct u_reduce<Op, float, true> {
  public:
    constexpr float operator()(const float *input, size_t input_stride,
                               size_t count, float init_value) noexcept {
        using binary_op_t =
            typename reduce_to_binary_type<Op>::template type<float, float>;
        binary_op_t op;
        if (count / 4) {
            float tmp[4];
            for (size_t i = 0; i < 4; i++) {
                tmp[i] = input[i * input_stride];
            }
            input += input_stride * 4;
            count -= 4;
            while (count / 4) {
                for (size_t i = 0; i < 4; i++) {
                    tmp[i] = op(tmp[i], input[i * input_stride]);
                }
                input += input_stride * 4;
                count -= 4;
            }

            tmp[0] = op(tmp[0], tmp[1]);
            tmp[2] = op(tmp[2], tmp[3]);
            tmp[0] = op(tmp[0], tmp[2]);
            init_value = op(init_value, tmp[0]);
        }

        for (size_t i = 0; i < count; i++) {
            init_value = op(init_value, *input);
            input += input_stride;
        }
        return init_value;
    }
};
} // namespace nncase::ntt::ukernels
