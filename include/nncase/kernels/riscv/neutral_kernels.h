/* Copyright 2019-2020 Canaan Inc.
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
#include "../kernel_utils.h"
#include <cmath>
#include <runtime/runtime_op_utility.h>
#include <xtl/xspan.hpp>

namespace nncase
{
namespace kernels
{
    namespace neutral
    {
        template <class TQ>
        void riscv_dequantize(const TQ *CXX_RESTRICT input, float *CXX_RESTRICT output, size_t count, const quant_param_t &param)
        {
            float scale = 1.f / param.scale;
            float zero = -param.zero_point * scale;

            for (size_t i = 0; i < count / 2; i++)
            {
                // handwritten pipeline for in order CPU
                auto in1_q = input[i * 2];
                auto in2_q = input[i * 2 + 1];
                auto in1 = (float)in1_q;
                auto in2 = (float)in2_q;
                auto out1 = in1 * scale + zero;
                auto out2 = in2 * scale + zero;

                output[i * 2] = out1;
                output[i * 2 + 1] = out2;
            }

            if (count % 2)
                output[count - 1] = input[count - 1] * scale + zero;
        }

        template <class TQ>
        void riscv_quantize(const float *CXX_RESTRICT input, TQ *CXX_RESTRICT output, size_t count, const quant_param_t &param)
        {
            float scale = param.scale;
            float zero = param.zero_point;

            for (size_t i = 0; i < count / 2; i++)
            {
                auto in1 = input[i * 2];
                auto in2 = input[i * 2 + 1];
                in1 = in1 * scale + zero;
                in2 = in2 * scale + zero;
                int32_t out1, out2;
                asm volatile("fcvt.w.s %0, %1, rne"
                             : "=r"(out1)
                             : "f"(in1));
                asm volatile("fcvt.w.s %0, %1, rne"
                             : "=r"(out2)
                             : "f"(in2));

                output[i * 2] = std::clamp(out1, (int32_t)std::numeric_limits<TQ>::lowest(), (int32_t)std::numeric_limits<TQ>::max());
                output[i * 2 + 1] = std::clamp(out2, (int32_t)std::numeric_limits<TQ>::lowest(), (int32_t)std::numeric_limits<TQ>::max());
            }

            if (count % 2)
            {
                auto in = (int32_t)roundf(input[count - 1] * scale + zero);
                output[count - 1] = std::clamp(in, (int32_t)std::numeric_limits<TQ>::lowest(), (int32_t)std::numeric_limits<TQ>::max());
            }
        }
    }
}
}
