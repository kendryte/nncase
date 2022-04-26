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
#include "runtime_types.h"
#include <nncase/kernels/kernel_context.h>

BEGIN_NS_NNCASE_KERNELS_CPU_REF

NNCASE_API result<void> nnil_unary_method(const float *input, float *output, size_t count, gsl::span<const gsl::byte> body, kernel_context &context) noexcept;

END_NS_NNCASE_KERNELS_CPU_REF
