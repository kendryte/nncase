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

#include "distributed.h"
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace nncase::ntt;
using namespace nncase::ntt::distributed;

namespace nncase::ntt {

// static nncase::ntt::runtime::timer_record
//     timer_records[CHIP_COUNTER][BLOCK_COUNTER][THREAD_COUNTER];

// auto_profiler, start timing and end timing
class auto_profiler {
  public:
    inline uint64_t get_current_time() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    }

    auto_profiler(std::string_view function_name)
        : cid_(program_id<topology::chip>()),
          bid_(program_id<topology::block>()),
          tid_(program_id<topology::thread>()) {

        en_profiler_ = get_profiler_option();
        if (en_profiler_) {

            // [[maybe_unused]] auto temp = program_ids();
            // std::cout << "cid: " << cid_ << ", bid: " << bid_
            //           << ", tid: " << tid_ << std::endl;
            timer_storage_ = get_timer_record();
            function_name_ = function_name;
            start_time_ = get_current_time();
        }
    }

    ~auto_profiler() {
        if (en_profiler_) {
            timer_storage_->set_id({cid_, bid_, tid_});
            end_time_ = get_current_time();
            timer_storage_->set_time(function_name_, start_time_, end_time_);
        }
    }

  private:
    std::string_view function_name_;
    uint64_t start_time_;
    uint64_t end_time_;
    int cid_;
    int bid_;
    int tid_;
    nncase::ntt::runtime::timer_record *timer_storage_;
    bool en_profiler_;

    inline bool get_profiler_option() noexcept {
        return runtime::cpu_thread_context_t::current().en_profiler;
    }

    inline nncase::ntt::runtime::timer_record *get_timer_record() noexcept {
        return runtime::cpu_thread_context_t::current().timer_records;
    }
};

} // namespace nncase::ntt