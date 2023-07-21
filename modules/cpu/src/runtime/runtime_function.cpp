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
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;

cpu_runtime_module &cpu_runtime_function::module() const noexcept {
    return static_cast<cpu_runtime_module &>(runtime_function::module());
}

result<void> cpu_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    text_ = context.module_init_context().section(".text").subspan(
        context.header().entrypoint, context.header().text_size);

    return ok();
}

result<value_t>
cpu_runtime_function::invoke_core(NNCASE_UNUSED gsl::span<value_t> parameters,
                                  value_t return_value) noexcept {
    // try_(preprocess_inputs());

    // vk::SubmitInfo si({}, {}, cmd_buffer_, {});
    // try_(vk::to_result(module().compute_queue().submit(si)));
    // try_(vk::to_result(module().compute_queue().waitIdle()));
    // try_(vk::to_result(module().device().waitIdle()));

    // assert(buffer_refs_.empty());
    // assert(buffer_copies_.empty());
    // assert(buffer_barriers_.empty());

    // try_(postprocess_outputs());
    return ok(return_value);
}