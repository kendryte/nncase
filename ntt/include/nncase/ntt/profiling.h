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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace nncase::ntt;
using namespace nncase::ntt::distributed;

namespace nncase::ntt {

#define CHIP_COUNTER 2
#define BLOCK_COUNTER 3
#define THREAD_COUNTER 4

class ntt_profiler {
  public:
    struct instance_id {
        int cid = -1;
        int bid = -1;
        int tid = -1;
    };

    struct call_instance {
        uint64_t start_time;
        uint64_t end_time;
    };

    struct function_stats {
        uint64_t call_count = 0;
        uint64_t total_time = 0;
        std::vector<call_instance> calls;
    };

    static ntt_profiler &get_instance() {
        static ntt_profiler instances[CHIP_COUNTER][BLOCK_COUNTER]
                                     [THREAD_COUNTER];

        auto cid = program_id<topology::chip>();
        auto bid = program_id<topology::block>();
        auto tid = program_id<topology::thread>();

        instances[cid][bid][tid].instance_id_ = {(int)cid, (int)bid, (int)tid};

        return instances[cid][bid][tid];
    }

    bool is_valid() const {
        return instance_id_.cid != -1 && instance_id_.bid != -1 &&
               instance_id_.tid != -1;
    }

    uint64_t start_timing() { return get_current_time(); }

    void end_timing(std::string_view function_name, uint64_t start_time) {
        uint64_t end_time = get_current_time();
        auto &stats = function_stats_[function_name];
        stats.calls.push_back({start_time, end_time});
        stats.call_count++;
        stats.total_time += end_time - start_time;
    }

    // print statistics
    void console_print() const {

        if (is_valid()) {
            uint64_t total_time = 0;
            for (const auto &[name, stats] : function_stats_) {
                total_time += stats.total_time;
            }

            std::cout << "\033[34m\n"
                      << "Core Id:" << instance_id_.cid
                      << ", Block Id:" << instance_id_.bid
                      << ", Thread Id:" << instance_id_.tid << "\033[0m\n";
            std::cout << "\033[34mStatistics for NTT kernels. Total time: "
                      << total_time << " microseconds. \033[0m\n";
            for (const auto &[name, stats] : function_stats_) {
                std::cout << "Function: " << name << "\n";
                std::cout << "\tCalls: " << stats.call_count << "\n";
                std::cout << "\tTotal time: " << stats.total_time
                          << " microseconds\n";
                std::cout << "\tTime Ratio: " << std::fixed
                          << std::setprecision(2)
                          << static_cast<double>(stats.total_time) /
                                 static_cast<double>(total_time)
                          << std::endl;
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

    void csv_print(std::string_view filename) const {
        if (is_valid()) {
            std::ofstream csv_file(filename.data());
            if (!csv_file.is_open()) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            // Write CSV headers
            csv_file
                << "Function,Calls,Total Time (microseconds),Time Ratio,Call "
                   "Index,Start Time (microseconds),End Time "
                   "(microseconds),Duration (microseconds)\n";

            uint64_t total_time = 0;
            for (const auto &[name, stats] : function_stats_) {
                total_time += stats.total_time;
            }

            for (const auto &[name, stats] : function_stats_) {
                double time_ratio = static_cast<double>(stats.total_time) /
                                    static_cast<double>(total_time);
                for (size_t i = 0; i < stats.calls.size(); ++i) {
                    const auto &call = stats.calls[i];
                    uint64_t duration = call.end_time - call.start_time;

                    // Write each call's data to the CSV file
                    csv_file << name << ",";
                    csv_file << stats.call_count << ",";
                    csv_file << stats.total_time << ",";
                    csv_file << std::fixed << std::setprecision(2) << time_ratio
                             << ",";
                    csv_file << i << ",";
                    csv_file << call.start_time << ",";
                    csv_file << call.end_time << ",";
                    csv_file << duration << "\n";
                }
            }

            csv_file.close();
        }
    }

    void markdown_print(std::string_view filename) const {

        if (is_valid()) {
            std::ofstream md_file(filename.data());
            if (!md_file.is_open()) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            uint64_t total_time = 0;
            for (const auto &[name, stats] : function_stats_) {
                total_time += stats.total_time;
            }

            // Write the header of the Markdown file
            md_file << "# NTT Kernel Statistics\n\n";
            md_file << "Total time: **" << total_time << " microseconds**\n\n";

            for (const auto &[name, stats] : function_stats_) {
                md_file << "## Function: " << name << "\n";
                md_file << "- Calls: **" << stats.call_count << "**\n";
                md_file << "- Total Time: **" << stats.total_time
                        << " microseconds**\n";
                md_file << "- Time Ratio: **" << std::fixed
                        << std::setprecision(2)
                        << static_cast<double>(stats.total_time) /
                               static_cast<double>(total_time)
                        << "**\n\n";

                // Write a table for the function calls
                md_file
                    << "| Call Index | Start Time (microseconds) | End Time "
                       "(microseconds) | Duration (microseconds) |\n";
                md_file
                    << "|------------|----------------------------|------------"
                       "--------------|-------------------------|\n";

                for (size_t i = 0; i < stats.calls.size(); ++i) {
                    const auto &call = stats.calls[i];
                    uint64_t duration = call.end_time - call.start_time;

                    md_file << "| " << i << " | " << call.start_time << " | "
                            << call.end_time << " | " << duration << " |\n";
                }

                md_file << "\n";
            }

            md_file << "\n*Note*: The `Time Ratio` is the fraction of the "
                       "total time "
                       "taken by each function.\n";

            md_file.close();
        }
    }

    void json_print(std::string_view filename) const {

        if (is_valid()) {
            std::ofstream json_file(filename.data());
            if (!json_file.is_open()) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            auto pid = instance_id_.cid * BLOCK_COUNTER + instance_id_.bid;
            auto tid = instance_id_.tid;
            json_file << "[\n";
            json_file << "  {\n"
                      << "    \"name\": \"process_name\",\n"
                      << "    \"ph\": \"M\",\n"
                      << "    \"pid\": " << pid << ",\n"
                      << "    \"tid\": " << tid << ",\n"
                      << "    \"args\": {\n"
                      << "      \"name\": \"cid*B+bid\"\n"
                      << "    }\n"
                      << "  },\n"
                      << "  {\n"
                      << "    \"name\": \"thread_name\",\n"
                      << "    \"ph\": \"M\",\n"
                      << "    \"pid\": " << pid << ",\n"
                      << "    \"tid\": " << tid << ",\n"
                      << "    \"args\": {\n"
                      << "      \"name\": \"tid\"\n"
                      << "    }\n"
                      << "  }";

            for (const auto &[name, stats] : function_stats_) {
                for (const auto &call : stats.calls) {

                    json_file << ",\n";
                    json_file << "  {\n";
                    json_file << "    \"name\": \"" << name << "\",\n";
                    json_file << "    \"ph\": \"B\",\n";
                    json_file << "    \"ts\": " << call.start_time << ",\n";
                    json_file << "    \"pid\": " << pid << ",\n";
                    json_file << "    \"tid\": " << tid << "\n";
                    json_file << "  }";

                    json_file << ",\n";
                    json_file << "  {\n";
                    json_file << "    \"name\": \"" << name << "\",\n";
                    json_file << "    \"ph\": \"E\",\n";
                    json_file << "    \"ts\": " << call.end_time << ",\n";
                    json_file << "    \"pid\": " << pid << ",\n";
                    json_file << "    \"tid\": " << tid << "\n";
                    json_file << "  }";
                }
            }
            json_file << "\n]\n";
            json_file.close();
        }
    }

  private:
    ntt_profiler() = default;

    ~ntt_profiler() {
        console_print();
        markdown_print("ntt_profiler.md");
        csv_print("ntt_profiler.csv");
        json_print("ntt_profiler.json");
    }

    uint64_t get_current_time() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    }

    std::unordered_map<std::string_view, function_stats> function_stats_;

    instance_id instance_id_ = {-1, -1, -1};
};

// auto_profiler, start timing and end timing
class auto_profiler {
  public:
    auto_profiler(std::string_view function_name) {

        en_profiler_ = get_profiler_option();
        if (en_profiler_) {
            function_name_ = function_name,
            start_time_ = ntt_profiler::get_instance().start_timing();
        }
    }

    ~auto_profiler() {
        if (en_profiler_) {
            ntt_profiler::get_instance().end_timing(function_name_,
                                                    start_time_);
        }
    }

  private:
    std::string_view function_name_;
    uint64_t start_time_;
    bool en_profiler_;
};

} // namespace nncase::ntt