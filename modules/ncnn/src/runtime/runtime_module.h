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
#include <nncase/runtime/ncnn/runtime_module.h>
#include <unordered_map>

BEGIN_NS_NNCASE_RT_MODULE(ncnn)

class ncnn_runtime_module : public runtime_module {
  public:
    kernels::kernel_context &kernel_context() noexcept;

    gsl::span<const gsl::byte> text() const noexcept { return text_; }
    gsl::span<const gsl::byte> rdata() const noexcept { return rdata_; }

  protected:
    result<void> initialize_before_functions(
        runtime_module_init_context &context) noexcept override;
    result<std::unique_ptr<runtime_function>>
    create_function() noexcept override;

  private:
    gsl::span<const gsl::byte> text_;
    gsl::span<const gsl::byte> rdata_;
    host_buffer_t text_storage_;
    host_buffer_t rdata_storage_;
};

END_NS_NNCASE_RT_MODULE
