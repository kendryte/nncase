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

#include <algorithm>
#include <map>
#include <nncase/runtime/stackvm/op_profile.h>
#include <unordered_map>
#include <vector>

std::vector<std::tuple<std::string, uint8_t, double, double>>
    op_profile::op_timing_;

void op_profile::print() {
    std::map<std::string, double> op_timing;
    std::unordered_map<std::string, size_t> op_count;

    std::cout << "stack OPs timeline" << std::endl;
    std::cout << "|" << std::setw(24) << std::left << "stackvm tensor op"
              << "|" << std::setw(24) << std::left << "start timing(ms)"
              << "|" << std::setw(24) << std::left << "end timing(ms)"
              << "|" << std::setw(24) << std::left << "cast(ms)"
              << "|" << std::endl;

    std::cout << "|" << std::setw(24) << std::left << "---"
              << "|" << std::setw(24) << std::left << "---"
              << "|" << std::setw(24) << std::left << "---"
              << "|" << std::setw(24) << std::left << "---"
              << "|" << std::endl;
    double init_timing = -1;
    for (auto &&[op_name, op_type, begin, end] : op_timing_) {
        if (init_timing == -1) {
            init_timing = begin;
        }
        auto cast_time = end - begin;
        if (op_timing.find(op_name) == op_timing.end()) {
            op_timing.emplace(op_name, cast_time);
            op_count.emplace(op_name, 1);
        } else {
            op_timing[op_name] += cast_time;
            op_count[op_name] += 1;
        }
        if (op_type == (uint8_t)nncase::runtime::stackvm::opcode_t::EXTCALL ||
            op_type == (uint8_t)nncase::runtime::stackvm::opcode_t::TENSOR ||
            op_type == (uint8_t)nncase::runtime::stackvm::opcode_t::CUSCALL)
            std::cout << "|" << std::setw(24) << std::left << op_name << "|"
                      << std::setw(24) << begin - init_timing << "|"
                      << std::setw(24) << end - init_timing << "|"
                      << std::setw(24) << end - begin << "|" << std::endl;
    }

    double total = 0.f;
    std::vector<std::pair<std::string, double>> v;
    for (auto e : op_timing) {
        total += e.second;
        v.push_back(e);
    }
    std::cout << std::endl;

    std::sort(
        v.begin(), v.end(),
        [=](std::pair<std::string, double> &a,
            std::pair<std::string, double> &b) { return a.second > b.second; });

    std::cout << "stackvm OPs profile" << std::endl;
    std::cout << "|" << std::setw(24) << std::left << "stackvm tensor op"
              << "|" << std::setw(6) << std::left << "count"
              << "|" << std::setw(12) << std::left << "timing(ms)"
              << "|" << std::setw(12) << std::left << "percent(%)"
              << "|" << std::endl;

    std::cout << "|" << std::setw(24) << std::left << "---"
              << "|" << std::setw(6) << std::left << "---"
              << "|" << std::setw(12) << std::left << "---"
              << "|" << std::setw(12) << std::left << "---"
              << "|" << std::endl;

    auto total_count = 0;
    for (auto e : v) {
        auto count = op_count[e.first];
        std::cout << "|" << std::setw(24) << std::left << e.first << "|"
                  << std::setw(6) << count << "|" << std::setw(12) << std::left
                  << e.second << "|" << std::setw(12) << std::left
                  << e.second / total * 100 << "|" << std::endl;
        total_count += count;
    }

    std::cout << "|" << std::setw(24) << std::left << "total"
              << "|" << std::setw(6) << std::left << total_count << "|"
              << std::setw(12) << std::left << total << "|" << std::setw(12)
              << std::left << total / total * 100 << "|" << std::endl
              << std::endl;
    op_timing_.clear();
    op_timing.clear();
}