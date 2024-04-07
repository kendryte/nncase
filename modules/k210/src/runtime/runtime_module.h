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
#include <nncase/runtime/k210/runtime_module.h>
#include <nncase/runtime/k210/runtime_types.h>

BEGIN_NS_NNCASE_RT_MODULE(k210)

class k210_runtime_module : public runtime_module {
  public:
    std::span<std::byte> data() const noexcept;
    std::span<const std::byte> rdata() const noexcept;
    std::span<std::byte> kpu_ram() noexcept;

#if !NNCASE_SIMULATOR
    uint32_t dma_ch() const noexcept { return dma_ch_; }
#endif

  protected:
    result<void> initialize_before_functions(
        runtime_module_init_context &context) noexcept override;
    result<std::unique_ptr<runtime_function>>
    create_function() noexcept override;

  private:
    std::unique_ptr<std::byte[]> data_;
    std::span<const std::byte> rdata_;
    std::span<const std::byte> text_;
#ifdef NNCASE_SIMULATOR
    std::array<std::byte, KPU_RAM_SIZE> kpu_ram_;
#else
    uint32_t dma_ch_;
#endif
};

END_NS_NNCASE_RT_MODULE
