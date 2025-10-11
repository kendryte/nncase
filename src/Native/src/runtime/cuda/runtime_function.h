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
#include "nncase/ntt/profiling.h"
#include "runtime_module.h"
#include <nncase/ntt/arch/cuda/runtime.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_function.h>
#include <nncase/tensor.h>
#include <vector>

BEGIN_NS_NNCASE_RT_MODULE(cuda)

class cuda_runtime_function final : public runtime_function {
  public:
    static constexpr size_t default_profile_record_count = 10000;

    cuda_runtime_function(runtime_module &rt_module);
    virtual ~cuda_runtime_function();

    cuda_runtime_module &module() const noexcept;

    const std::span<std::byte>
    thread_local_data(size_t block_id) const noexcept {
        auto &local_data = thread_local_datas_[block_id];
        auto mapped_local_data =
            local_data->map(map_read_write).expect("Failed to map local data");
        return mapped_local_data.buffer();
    }

    const std::span<std::byte>
    block_local_data(size_t block_id) const noexcept {
        auto &local_data = block_local_datas_[block_id];
        auto mapped_local_data =
            local_data->map(map_read_write).expect("Failed to map local data");
        return mapped_local_data.buffer();
    }

    const std::span<ntt::runtime::profile_record>
    thread_local_profile_records(size_t block_id) noexcept {
        return profile_records_[block_id];
    }

    const std::span<uint32_t>
    thread_local_profile_record_counts(size_t block_id) noexcept {
        return profile_record_counts_[block_id];
    }

  protected:
    result<void>
    initialize_core(runtime_function_init_context &context) noexcept override;
    result<value_t> invoke_core(std::span<value_t> parameters,
                                value_t return_value) noexcept override;

  private:
    result<void> run(std::byte *output_data) noexcept;
    result<tensor> create_output_tensor(size_t output_id,
                                        std::span<value_t> parameters,
                                        std::byte *output_data) noexcept;

  private:
    block_entry_t block_entry_;
    std::vector<host_buffer_t> thread_local_datas_;
    std::vector<host_buffer_t> block_local_datas_;
    host_buffer_t output_buffer_;
    std::vector<ntt::runtime::thread_inout_desc> input_descs_;
    std::vector<ntt::runtime::thread_inout_desc> output_descs_;
    std::vector<dims_t> output_shapes_;
    std::vector<dims_t> output_strides_;

    std::vector<std::vector<ntt::runtime::profile_record>> profile_records_;
    std::vector<std::vector<uint32_t>> profile_record_counts_;
};

END_NS_NNCASE_RT_MODULE
