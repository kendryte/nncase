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
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>

extern "C" {
double get_ms_time();
}

class op_profile {
  public:
    op_profile(const std::string &op_type = "op_profile") : op_type_(op_type) {
        begin_ = get_ms_time();
    }

    ~op_profile() {
        end_ = get_ms_time();
        auto cast_time = end_ - begin_;
        if (op_type_ == "EXTCALL") {
            std::cout << "extcall time:" << cast_time << std::endl;
        }
        if (op_timing_.find(op_type_) == op_timing_.end()) {
            op_timing_.emplace(op_type_, cast_time);
            op_count_.emplace(op_type_, 1);
        } else {
            op_timing_[op_type_] += cast_time;
            op_count_[op_type_] += 1;
        }
    }

    static void print();

  public:
    static std::unordered_map<std::string, double> op_timing_;
    static std::map<std::string, size_t> op_count_;

  private:
    double begin_;
    double end_;
    std::string op_type_;
};