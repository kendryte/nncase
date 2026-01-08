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
#include "opcode.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

using namespace nncase::runtime::stackvm;

extern "C" {
double get_ms_time();
}

class op_profile {
  public:
    op_profile(opcode_t opcode, uint8_t enable_profiling = 1)
        : active_(enable_profiling) {
        if (active_) {
            op_type_ = (uint8_t)opcode;
            op_name_ = to_string(opcode);
            begin_ = get_time();
        }
    }

    op_profile(opcode_t opcode, tensor_function_t tensor_funct,
               uint8_t enable_profiling = 1)
        : active_(enable_profiling) {
        if (active_) {
            op_type_ = (uint8_t)opcode;
            op_name_ = to_string(tensor_funct);
            begin_ = get_time();
        }
    }

    ~op_profile() {
        if (active_) {
            end_ = get_time();
            op_timing_.push_back(
                std::make_tuple(op_name_, op_type_, begin_, end_));
        }
    }

    static void print();

  public:
    static std::vector<std::tuple<std::string, uint8_t, double, double>>
        op_timing_;

  private:
    double get_time() {
#if defined(NNCASE_BAREMETAL)
        return get_ms_time();
#elif defined(LINUX_RUNTIME)
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<
            std::chrono::duration<double, std::milli>>(now.time_since_epoch());
        return duration.count();
#else
        return (double)clock() / 1000;
#endif
    }

  private:
    double begin_;
    double end_;
    std::string op_name_;
    uint8_t op_type_;
    uint8_t active_;
};