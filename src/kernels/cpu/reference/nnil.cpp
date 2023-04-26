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
#include <nncase/kernels/cpu/reference/nnil.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/nnil.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

result<void> reference::nnil_unary_method(const float *input, float *output, size_t count, gsl::span<const gsl::byte> body, NNCASE_UNUSED kernel_context &context) noexcept
{
    for (size_t i = 0; i < count; i++)
    {
        nnil_evalstack stack;
        span_reader sr(body);
        nnil_reader reader(sr);
        bool ret = false;

        while (reader.avail() && !ret)
        {
            auto op = reader.next();
            switch (op.opcode)
            {
            case nnil_nop:
                break;
            case nnil_dup:
                stack.dup();
                break;
            case nnil_pop:
                stack.pop();
                break;
            case nnil_lda_0:
                stack.push(input[i]);
                break;
            case nnil_ldc_r4_0:
                stack.push(0.f);
                break;
            case nnil_ldc_r4_1:
                stack.push(1.f);
                break;
            case nnil_ldc_r4:
                stack.push(op.ldc_r4.r4);
                break;
            case nnil_abs:
                stack.push(fabsf(stack.pop()));
                break;
            case nnil_acos:
                stack.push(acosf(stack.pop()));
                break;
            case nnil_asin:
                stack.push(asin(stack.pop()));
                break;
            case nnil_ceil:
                stack.push(ceilf(stack.pop()));
                break;
            case nnil_cos:
                stack.push(cosf(stack.pop()));
                break;
            case nnil_exp:
                stack.push(expf(stack.pop()));
                break;
            case nnil_floor:
                stack.push(floorf(stack.pop()));
                break;
            case nnil_erf:
                stack.push(erff(stack.pop()));
                break;
            case nnil_log:
                stack.push(logf(stack.pop()));
                break;
            case nnil_logical_not:
                stack.push(!(stack.pop()));
                break;
            case nnil_neg:
                stack.push(-stack.pop());
                break;
            case nnil_round:
                stack.push(roundf(stack.pop()));
                break;
            case nnil_rsqrt:
                stack.push(1.f / sqrtf(stack.pop()));
                break;
            case nnil_sign:
            {
                auto val = stack.pop();
                stack.push((0 < val) - (val < 0));
                break;
            }
            case nnil_sin:
                stack.push(sinf(stack.pop()));
                break;
            case nnil_sqrt:
                stack.push(sqrtf(stack.pop()));
                break;
            case nnil_square:
            {
                auto v = stack.pop();
                stack.push(v * v);
                break;
            }
            case nnil_tanh:
                stack.push(tanhf(stack.pop()));
                break;
            case nnil_add:
            {
                auto b = stack.pop();
                auto a = stack.pop();
                stack.push(a + b);
                break;
            }
            case nnil_sub:
            {
                auto b = stack.pop();
                auto a = stack.pop();
                stack.push(a - b);
                break;
            }
            case nnil_mul:
            {
                auto b = stack.pop();
                auto a = stack.pop();
                stack.push(a * b);
                break;
            }
            case nnil_div:
            {
                auto b = stack.pop();
                auto a = stack.pop();
                stack.push(a / b);
                break;
            }
            case nnil_min:
            {
                auto b = stack.pop();
                auto a = stack.pop();
                stack.push(std::min(a, b));
                break;
            }
            case nnil_max:
            {
                auto b = stack.pop();
                auto a = stack.pop();
                stack.push(std::max(a, b));
                break;
            }
            case nnil_pow:
            {
                auto b = stack.pop();
                auto a = stack.pop();
                stack.push(std::pow(a, b));
                break;
            }
            case nnil_logical_and:
            {
                auto b = stack.pop();
                auto a = stack.pop();
                stack.push(a && b);
                break;
            }
            case nnil_clamp:
            {
                auto high = stack.pop();
                auto low = stack.pop();
                auto v = stack.pop();
                stack.push(clamp(v, low, high));
                break;
            }
            case nnil_ret:
                output[i] = stack.pop();
                ret = true;
                break;
            default:
                return err(nncase_errc::nnil_illegal_instruction);
            }
        }
    }

    return ok();
}
