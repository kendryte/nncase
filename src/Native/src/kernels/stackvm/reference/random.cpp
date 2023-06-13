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
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <random>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;

template <typename T>
result<void> random_normal_impl(T *output, gsl::span<const size_t> out_shape, float mean,
                                float std, float seed) noexcept {
    std::default_random_engine engine(seed);
    std::normal_distribution<T> dis(mean, std);
    std::generate_n(output, compute_size(out_shape),
                    [&] { return dis(engine); });

    return ok();
}

template <typename T>
result<void> random_uniform_impl(T *output, gsl::span<const size_t> out_shape, float low,
                                 float high, float seed) noexcept {
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<T> dis(low, high);
    std::generate_n(output, compute_size(out_shape),
                    [&] { return dis(engine); });

    return ok();
}

result<void> nncase::kernels::stackvm::reference::random_normal(
    typecode_t type, gsl::byte *output, gsl::span<const size_t> out_shape, float mean,
    float std, float seed) noexcept {
    if (type != dt_float32) {
        return err(nncase_errc::datatype_mismatch);
    }
    return random_normal_impl(OUT_CAST(float, output), out_shape, mean, std,
                              seed);
}

result<void> nncase::kernels::stackvm::reference::random_uniform(
    typecode_t type, gsl::byte *output, gsl::span<const size_t> out_shape, float low,
    float high, float seed) noexcept {
    if (type != dt_float32) {
        return err(nncase_errc::datatype_mismatch);
    }
    return random_uniform_impl(OUT_CAST(float, output), out_shape, low, high,
                               seed);
}