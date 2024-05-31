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
#include "runtime_module.h"
#include <nncase/runtime/k210/op_reader.h>
#include <nncase/runtime/k210/runtime_types.h>
#include <nncase/runtime/runtime_function.h>

BEGIN_NS_NNCASE_RT_MODULE(k210)

class k210_runtime_function : public runtime_function, private op_visitor {
  public:
    using runtime_function::runtime_function;

    k210_runtime_module &module() const noexcept;

  protected:
    result<void>
    initialize_core(runtime_function_init_context &context) noexcept override;
    result<runtime_tensor>
    allocate_input_tensor(size_t index) noexcept override;
    result<runtime_tensor>
    allocate_output_tensor(size_t index) noexcept override;
    result<void> validate_input_tensor(size_t index,
                                       runtime_tensor tensor) noexcept override;
    result<void>
    validate_output_tensor(size_t index,
                           runtime_tensor tensor) noexcept override;
    result<void> invoke_core() noexcept override;

    using op_visitor::visit;
    result<void> visit(const kpu_conv2d_options &op) noexcept override;
    result<void> visit(const kpu_download_options &op) noexcept override;
    result<void> visit(const kpu_upload_options &op) noexcept override;
    result<void> visit(const copy_options &op) noexcept override;

  private:
    result<std::span<std::byte>> memory_at(const memory_range &mrange) noexcept;

  private:
    std::span<const std::byte> text_;
};

END_NS_NNCASE_RT_MODULE
