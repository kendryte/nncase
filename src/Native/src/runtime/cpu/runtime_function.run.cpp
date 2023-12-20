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
#include "runtime_function.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/type_serializer.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;

namespace {
nncase_runtime_cpu_mt_t nncase_cpu_mt_ = {
    .acosf = acosf,
    .acoshf = acoshf,
    .asinf = asinf,
    .asinhf = asinhf,
    .copysignf = copysignf,
    .cosf = cosf,
    .coshf = coshf,
    .expf = expf,
    .fmodf = fmodf,
    .logf = logf,
    .nearbyintf = nearbyintf,
    .powf = powf,
    .sinf = sinf,
    .sinhf = sinhf,
    .tanhf = tanhf,

#if !defined(WIN32)
    .memcpy = memcpy,
#endif
};
} // namespace

result<void> cpu_runtime_function::run(gsl::span<gsl::byte *> params) noexcept {
    kernel_entry_(&nncase_cpu_mt_, params.data(), module().rdata().data());
    return ok();
}
