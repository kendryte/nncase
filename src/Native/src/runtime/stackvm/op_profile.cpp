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
#include <nncase/runtime/stackvm/op_profile.h>
#include <vector>

#ifdef NNCASE_BAREMETAL
double get_ms_time();
#else
double get_ms_time() { return (double)clock() / 1000; }
#endif

std::vector<std::tuple<std::string, double, double>> op_profile::op_timing_;

void op_profile::print() {

    std::unordered_map<std::string, double> op_timing;
    std::unordered_map<std::string, size_t> op_count;

    std::cout << "stack OPs timeline" << std::endl;
    std::cout << "|" << std::setw(24) << std::left << "stackvm tensor op"
          << "|" << std::setw(12) << std::left << "start timing(ms)"
          << "|" << std::setw(12) << std::left << "end timing(ms)"
          << "|" << std::endl;

   std::cout << "|" << std::setw(24) << std::left << "---"
              << "|" << std::setw(12) << std::left << "---"
              << "|" << std::setw(12) << std::left << "---"
              << "|" << std::endl;

    for(auto &&[op_type, begin, end] : op_timing_) {
      auto cast_time = end - begin;
      if (op_timing.find(op_type) == op_timing.end()) {
          op_timing.emplace(op_type, cast_time);
          op_count.emplace(op_type, 1);
      } else {
          op_timing[op_type] += cast_time;
          op_count[op_type] += 1;
      }
      std::cout << "|" << std::setw(24) << std::left << op_type
                << "|" << std::setw(12) << begin
                << "|" << std::setw(12) << end 
                << "|" << std::endl;
    }

    double total = 0.f;
    std::vector<std::pair<std::string, double>> v;
    for (auto e : op_timing) {
        total += e.second;
        v.push_back(e);
    }

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

    op_timing.clear();
}