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
#include <cstdint>
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/cpu/runtime_module.h>
#include <unordered_map>

#ifdef __APPLE__
#include <pthread.h>
#endif

BEGIN_NS_NNCASE_RT_MODULE(cpu)

class cpu_runtime_module : public runtime_module {
  public:
    cpu_runtime_module() noexcept;
    virtual ~cpu_runtime_module();

    kernels::kernel_context &kernel_context() noexcept;

    uint64_t tdim() const noexcept { return tdim_; }
    uint64_t bdim() const noexcept { return bdim_; }
    uint64_t cdim() const noexcept { return cdim_; }

    std::span<const std::byte> text() const noexcept { return text_; }
    std::span<const std::byte> rdata() const noexcept { return rdata_; }

    const std::span<const std::byte> local_rdata() const noexcept {
        return local_rdata_;
    }

    const uint64_t *local_rdata_header(size_t offset) const noexcept {
        return reinterpret_cast<const uint64_t *>(local_rdata_.data()) +
               offset * 2;
    }

    const std::span<const std::byte> local_rdata_content() const noexcept {
        return local_rdata_.subspan(cdim_ * bdim_ * tdim_ * 2 *
                                    sizeof(uint64_t));
    }

#ifdef __APPLE__
    pthread_key_t cpu_thread_context_key() const noexcept {
        return cpu_thread_context_key_;
    }
#endif

  protected:
    result<void> initialize_before_functions(
        runtime_module_init_context &context) noexcept override;
    result<std::unique_ptr<runtime_function>>
    create_function() noexcept override;

  private:
    uint64_t tdim_;
    uint64_t bdim_;
    uint64_t cdim_;
    std::span<const std::byte> text_;
    std::span<const std::byte> rdata_;
    std::span<const std::byte> local_rdata_;
    host_buffer_t text_storage_;
    host_buffer_t rdata_storage_;

#ifdef __APPLE__
    pthread_key_t cpu_thread_context_key_ = {};
#endif
};

END_NS_NNCASE_RT_MODULE
