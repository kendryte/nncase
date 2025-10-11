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
#include "compiler_defs.h"
#include <cstdint>

namespace nncase::ntt {
enum class profile_level { kernel, device };

namespace runtime {
struct profile_record {
    uint32_t function_id;
    uint64_t duration;
};

NTT_DEVICE bool is_profiling_enabled() noexcept;
NTT_DEVICE uint64_t get_profile_time() noexcept;
NTT_DEVICE void record_profile(profile_level level,
                               const profile_record &record) noexcept;
} // namespace runtime

class profile_scope {
  public:
    NTT_DEVICE
    profile_scope(uint32_t function_id,
                  profile_level level = profile_level::kernel) noexcept
        : enabled_(runtime::is_profiling_enabled()),
          function_id_(function_id),
          level_(level) {
        if (enabled_) {
            start_time_ = runtime::get_profile_time();
        }
    }

    NTT_DEVICE ~profile_scope() noexcept {
        if (enabled_) {
            auto duration = runtime::get_profile_time() - start_time_;
            runtime::profile_record record{function_id_, duration};
            runtime::record_profile(level_, record);
        }
    }

  private:
    bool enabled_;
    uint32_t function_id_;
    profile_level level_;
    uint64_t start_time_;
};
} // namespace nncase::ntt
