/* Copyright 2020 Canaan Inc.
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
#include <nncase/runtime/k210/op_reader.h>
#include <nncase/runtime/k210/runtime_module.h>
#include <nncase/runtime/k210/runtime_types.h>

BEGIN_NS_NNCASE_RT_K210

class k210_runtime_module : public runtime_module, private op_visitor
{
public:
protected:
    result<void> initialize_core(runtime_module_init_context &context) noexcept override;
    result<runtime_tensor> allocate_input_tensor(size_t index) noexcept override;
    result<runtime_tensor> allocate_output_tensor(size_t index) noexcept override;
    result<void> validate_input_tensor(size_t index, runtime_tensor tensor) noexcept override;
    result<void> validate_output_tensor(size_t index, runtime_tensor tensor) noexcept override;
    result<void> run_core() noexcept override;

    using op_visitor::visit;
    result<void> visit(const kpu_conv2d_options &op) noexcept override;
    result<void> visit(const kpu_download_options &op) noexcept override;
    result<void> visit(const kpu_upload_options &op) noexcept override;

private:
    result<gsl::span<gsl::byte>> memory_at(const memory_range &mrange) noexcept;

private:
    std::unique_ptr<gsl::byte[]> data_;
    gsl::span<const gsl::byte> rdata_;
    gsl::span<const gsl::byte> text_;
#ifdef NNCASE_SIMULATOR
    std::array<gsl::byte, KPU_RAM_SIZE> kpu_ram_;
#else
    uint32_t dma_ch_;
#endif
};

END_NS_NNCASE_RT_K210
