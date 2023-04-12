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

std::unordered_map<std::string, double> op_profile::op_timing_;
std::map<std::string, size_t> op_profile::op_count_;
void op_profile::print() {
    double total = 0.f;
    std::vector<std::pair<std::string, double>> v;
    for (auto e : op_timing_) {
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
        auto count = op_count_[e.first];
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
}