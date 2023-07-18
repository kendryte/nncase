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
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

extern "C" {
double get_ms_time();
}

class op_profile {
  public:
    op_profile(const std::string &op_name, uint8_t op_type)
        : op_name_(op_name), op_type_(op_type) {
        begin_ = get_ms_time();
    }

    ~op_profile() {
        end_ = get_ms_time();
        op_timing_.push_back(std::make_tuple(op_name_, op_type_, begin_, end_));
    }

    static void print();

  public:
    static std::vector<std::tuple<std::string, uint8_t, double, double>>
        op_timing_;

  private:
    double begin_;
    double end_;
    std::string op_name_;
    uint8_t op_type_;
};