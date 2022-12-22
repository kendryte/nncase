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
#include "../runtime_function.h"
#include <nncase/kernels/k210/k210_kernels.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

result<void>
k210_runtime_function::visit(const kpu_download_options &op) noexcept {
    try_var(input, memory_at(op.input));
    try_var(output, memory_at(op.output));

    return kernels::k210::kpu_download(
        reinterpret_cast<const uint8_t *>(input.data()),
        reinterpret_cast<uint8_t *>(output.data()), op.in_shape);
}
