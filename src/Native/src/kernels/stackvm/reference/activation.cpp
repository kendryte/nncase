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

#include "kernel_template.h"
#include <math.h>
#include <nncase/kernels/apply.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

UNARY_TEMPLATE(relu, std::max(0.f, (float)x))
UNARY_TEMPLATE(softsign, (float)x / (1.f + std::abs((float)x)))
UNARY_TEMPLATE(softplus, std::logf(1.f + std::expf((float)x)))
UNARY_TEMPLATE(sigmoid, 1.f / (1.f + std::expf((float)(-x))))
UNARY_TEMPLATE(hard_swish,
               (float)x *(float)std::max(
                   0.f,
                   (float)std::min(1.f, (float)(1.f / 6.f * (float)x + 0.5f))))
UNARY_TEMPLATE(erf, std::erff((float)x)) // for k510 toolchain
UNARY_WITH_MUL_TEMPLATE_V2(elu, alpha,
                           (float)x < 0.f
                               ? (float)(alpha * (std::expf((float)x) - 1))
                               : (float)x)
// FLOAT_UNARY_WITH_MUL_TEMPLATE(prelu, slope, x < 0 ? slope * x : x)
UNARY_WITH_MUL_TEMPLATE_V2(
    celu, alpha,
    std::max(0.f, (float)x) +
        std::min(0.f, (float)(alpha *(std::expf((float)x / alpha) - 1.f))))
UNARY_WITH_MUL_TEMPLATE_V2(leaky_relu, alpha,
                           (float)x < 0.f ? (float)(alpha * (float)x)
                                          : (float)x)
UNARY_WITH_MUL_TEMPLATE_V2(gelu, alpha,
                           (float)(0.5f * (alpha * (float)x) *
                                   (1.f + std::erff(alpha * (float)x /
                                                    std::sqrtf(2.f)))))
UNARY_WITH_MUL_TEMPLATE_V2(swish, alpha,
                           (float)x / (1.f + std::expf(-alpha * (float)x)))
ACTIVATION_TEMPLATE_V2(selu,
                       (float)x <= 0.f
                           ? (float)(gamma *
                                     (alpha * std::expf((float)x) - alpha))
                           : (float)((float)x * gamma),
                       alpha, gamma)
ACTIVATION_TEMPLATE_V2(hard_sigmoid,
                       std::max(0.f, std::min(1.f, (float)(x *alpha + gamma))),
                       alpha, gamma)