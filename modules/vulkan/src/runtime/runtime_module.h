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
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/vulkan/runtime_module.h>

BEGIN_NS_NNCASE_RT_MODULE(vulkan)

class vulkan_runtime_module : public runtime_module
{
public:
protected:
    result<void> initialize_core(runtime_module_init_context &context) noexcept override;
    result<runtime_tensor> allocate_input_tensor(size_t index) noexcept override;
    result<runtime_tensor> allocate_output_tensor(size_t index) noexcept override;
    result<void> validate_input_tensor(size_t index, runtime_tensor tensor) noexcept override;
    result<void> validate_output_tensor(size_t index, runtime_tensor tensor) noexcept override;
    result<void> run_core() noexcept override;

private:
    result<void> initialize_vulkan() noexcept;
private:
    std::unique_ptr<gsl::byte[]> data_;
    gsl::span<const gsl::byte> rdata_;
    gsl::span<const gsl::byte> text_;
};

END_NS_NNCASE_RT_MODULE
