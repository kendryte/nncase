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
#include "../../profiling.h"
#include "../../runtime.h"
#include <cstdarg>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <string>

#ifdef __APPLE__
#include <pthread.h>
#endif

namespace nncase::ntt::runtime {

struct record_id {
    int cid = -1;
    int bid = -1;
    int tid = -1;
};

class timer_record : public nncase::ntt::runtime::timer_record_base<record_id> {
  public:
    bool is_valid() const override {
        return instance_id_.cid != -1 && instance_id_.bid != -1 &&
               instance_id_.tid != -1;
    }

    void set_time(std::string_view function_name, uint64_t start_time,
                  uint64_t end_time) override {
        auto &stats = function_stats_[function_name];
        stats.calls.push_back({start_time, end_time});
        stats.call_count++;
        stats.total_time += end_time - start_time;
    }

    void set_level(std::string_view filename, profiling_level level) override {
        auto &stats = function_stats_[filename];
        stats.level = level;
    }

    // print statistics
    void console_print() const override {

        if (is_valid()) {

            std::cout << "\033[34m\n"
                      << "Core Id:" << instance_id_.cid
                      << ", Block Id:" << instance_id_.bid
                      << ", Thread Id:" << instance_id_.tid << "\033[0m\n";
            std::cout << "\033[34mStatistics for NTT kernels. \033[0m\n";
            for (const auto &[name, stats] : function_stats_) {
                std::cout << "Function: " << name << "\n";
                std::cout << "Level: " << ntt::runtime::to_string(stats.level)
                          << "\n";
                std::cout << "\tCalls: " << stats.call_count << "\n";
                std::cout << "\tTotal time: " << stats.total_time
                          << " microseconds\n";
                uint64_t call_count = 0;
                for (const auto &call : stats.calls) {
                    std::cout << "\t\t"
                              << "Call " << call_count++ << ": \n";
                    std::cout << "\t\tStart time: " << call.start_time
                              << " microseconds\n";
                    std::cout << "\t\tEnd time: " << call.end_time
                              << " microseconds\n";
                    std::cout
                        << "\t\tDuration: " << call.end_time - call.start_time
                        << " microseconds\n";
                }
            }
        }
    }

    void csv_print(std::string_view filename) const override {
        if (is_valid()) {
            std::ofstream csv_file(filename.data());
            if (!csv_file.is_open()) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            csv_file
                << "Core Id,Block Id,Thread Id,Function,Level,Calls,Total Time "
                   "(microseconds),Call Index,Start Time (microseconds),End "
                   "Time (microseconds),Duration (microseconds)\n";

            for (const auto &[name, stats] : function_stats_) {
                uint64_t call_count = 0;
                for (const auto &call : stats.calls) {
                    csv_file << instance_id_.cid << "," << instance_id_.bid
                             << "," << instance_id_.tid << "," << name << ","
                             << ntt::runtime::to_string(stats.level) << ","
                             << stats.call_count << "," << stats.total_time
                             << "," << call_count++ << "," << call.start_time
                             << "," << call.end_time << ","
                             << (call.end_time - call.start_time) << "\n";
                }
            }

            csv_file.close();
        }
    }

    void markdown_print(std::string_view filename) const override {

        if (is_valid()) {
            std::ofstream md_file(filename.data());
            if (!md_file.is_open()) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            md_file << "### Core Information\n";
            md_file << "| Core Id | Block Id | Thread Id |\n";
            md_file << "|---------|----------|-----------|\n";
            md_file << "| " << instance_id_.cid << " | " << instance_id_.bid
                    << " | " << instance_id_.tid << " |\n";

            md_file << "\n### NTT Kernels Statistics\n";

            for (const auto &[name, stats] : function_stats_) {
                md_file << "#### Function: " << name << "\n";
                md_file << "| Level | Calls | Total Time (microseconds) |\n";
                md_file << "|-------|-------|---------------------------|\n";
                md_file << "| " << ntt::runtime::to_string(stats.level) << " | "
                        << stats.call_count << " | " << stats.total_time
                        << " |\n";

                md_file << "\n**Call Details:**\n";
                md_file << "| Call Index | Start Time (microseconds) | End "
                           "Time (microseconds) | Duration (microseconds) |\n";
                md_file << "|------------|---------------------------|---------"
                           "----------------|-------------------------|\n";

                uint64_t call_count = 0;
                for (const auto &call : stats.calls) {
                    md_file << "| " << call_count++ << " | " << call.start_time
                            << " | " << call.end_time << " | "
                            << (call.end_time - call.start_time) << " |\n";
                }
                md_file << "\n";
            }

            md_file.close();
        }
    }

    void json_print(std::string_view filename) const override {

        if (is_valid()) {
            std::ofstream json_file(filename.data());
            if (!json_file.is_open()) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            std::string pid = "\"cid: " + std::to_string(instance_id_.cid) +
                              ", bid: " + std::to_string(instance_id_.bid) +
                              "\"";
            std::string tid =
                "\"tid: " + std::to_string(instance_id_.tid) + "\"";
            json_file << "[\n";

            bool first = true;
            for (const auto &[name, stats] : function_stats_) {
                for (const auto &call : stats.calls) {
                    if (stats.level == profiling_level::kernel) {
                        if (!first) {
                            json_file << ",\n";
                        }
                        first = false;
                        json_file << "  {\n";
                        json_file << "    \"name\": \"" << name << "\",\n";
                        json_file << "    \"ph\": \"X\",\n";
                        json_file << "    \"ts\": " << call.start_time << ",\n";
                        json_file << "    \"dur\": "
                                  << (call.end_time - call.start_time) << ",\n";
                        json_file << "    \"pid\": " << pid << ",\n";
                        json_file << "    \"tid\": " << tid << ",\n";
                        json_file << "    \"args\": { \"level:\":\""
                                  << ntt::runtime::to_string(stats.level)
                                  << " \"}\n";
                        json_file << "  }";
                    }
                }
            }

            for (const auto &[name, stats] : function_stats_) {
                for (const auto &call : stats.calls) {
                    if (stats.level == profiling_level::device) {
                        if (!first) {
                            json_file << ",\n";
                        }
                        first = false;
                        json_file << "  {\n";
                        json_file << "    \"name\": \"" << name << "\",\n";
                        json_file << "    \"ph\": \"X\",\n";
                        json_file << "    \"ts\": " << call.start_time << ",\n";
                        json_file << "    \"dur\": "
                                  << (call.end_time - call.start_time) << ",\n";
                        json_file << "    \"pid\": " << pid << ",\n";
                        json_file << "    \"tid\": " << tid << ",\n";
                        json_file << "    \"args\": { \"level:\":\""
                                  << ntt::runtime::to_string(stats.level)
                                  << " \"}\n";
                        json_file << "  }";
                    }
                }
            }

            json_file << "\n]\n";
            json_file.close();
        }
    }

    timer_record() = default;

    ~timer_record() {
        console_print();
        markdown_print("nncase_profiling.md");
        csv_print("nncase_profiling.csv");
        json_print("nncase_profiling.json");
    }

    void set_id(record_id id) override { instance_id_ = id; }
};

struct cpu_block_entry_params_t {
    size_t tdim;
    size_t bdim;
    size_t cdim;
    size_t bid;
    size_t cid;
    size_t cpu_id_offset;
    const thread_inout_desc *input_descs;
    thread_inout_desc *const output_descs;
    std::span<const std::byte> rdata;
    std::byte *output;
    uint8_t enable_profiling;
    timer_record *timer_records;
    const uint64_t *local_rdata_header;
    std::span<const std::byte> local_rdata;
    std::span<std::byte> block_local_data;
#ifdef __APPLE__
    pthread_key_t cpu_thread_context_key;
#endif
};

struct cpu_thread_context_t {
    size_t tid;
    size_t bid;
    size_t cid;
    timer_record *timer_records;
    uint8_t enable_profiling;

    static cpu_thread_context_t &current() noexcept;
};

extern size_t tdim;
extern size_t bdim;
extern size_t cdim;
} // namespace nncase::ntt::runtime

extern "C" NTT_RUNTIME_API void
block_entry(const nncase::ntt::runtime::cpu_block_entry_params_t &params);
using block_entry_t = decltype(block_entry) *;
