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

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace nncase::ntt {
class ntt_profiler {
  public:
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
        static ntt_profiler instance;
        return instance;
    }

    // 记录开始时间
    uint64_t start_timing() { return get_current_time(); }

    // 记录结束时间和计算持续时间
    void end_timing(const std::string &function_name, uint64_t start_time) {
        uint64_t end_time = get_current_time();
        auto &stats = function_stats_[function_name];
        stats.calls.push_back({start_time, end_time});
        stats.call_count++;
        stats.total_time += end_time - start_time;
    }

    // print statistics
    void console_print() const {
        uint64_t total_time = 0;
        for (const auto &[name, stats] : function_stats_) {
            total_time += stats.total_time;
        }

        std::cout << "\033[34m\nStatistics for NTT kernels. Total time: "
                  << total_time << " microseconds. \033[0m\n";
        for (const auto &[name, stats] : function_stats_) {
            std::cout << "Function: " << name << "\n";
            std::cout << "\tCalls: " << stats.call_count << "\n";
            std::cout << "\tTotal time: " << stats.total_time
                      << " microseconds\n";
            std::cout << "\tTime Ratio: " << std::fixed << std::setprecision(2)
                      << static_cast<double>(stats.total_time) /
                             static_cast<double>(total_time)
                      << std::endl;
            uint64_t call_count = 0;
            for (const auto &call : stats.calls) {
                std::cout << "\t\t" << "Call " << call_count++ << ": \n";
                std::cout << "\t\tStart time: " << call.start_time
                          << " microseconds\n";
                std::cout << "\t\tEnd time: " << call.end_time
                          << " microseconds\n";
                std::cout << "\t\tDuration: " << call.end_time - call.start_time
                          << " microseconds\n";
            }
        }
    }

    void csv_print(const std::string &filename) const {
        std::ofstream csv_file(filename);
        if (!csv_file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Write CSV headers
        csv_file << "Function,Calls,Total Time (microseconds),Time Ratio,Call "
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

    void markdown_print(const std::string &filename) const {

        std::ofstream md_file(filename);
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
            md_file << "- Time Ratio: **" << std::fixed << std::setprecision(2)
                    << static_cast<double>(stats.total_time) /
                           static_cast<double>(total_time)
                    << "**\n\n";

            // Write a table for the function calls
            md_file << "| Call Index | Start Time (microseconds) | End Time "
                       "(microseconds) | Duration (microseconds) |\n";
            md_file << "|------------|----------------------------|------------"
                       "--------------|-------------------------|\n";

            for (size_t i = 0; i < stats.calls.size(); ++i) {
                const auto &call = stats.calls[i];
                uint64_t duration = call.end_time - call.start_time;

                md_file << "| " << i << " | " << call.start_time << " | "
                        << call.end_time << " | " << duration << " |\n";
            }

            md_file << "\n";
        }

        md_file
            << "\n*Note*: The `Time Ratio` is the fraction of the total time "
               "taken by each function.\n";

        md_file.close();
    }

    void json_print(const std::string &filename) const {

        std::ofstream json_file(filename);
        if (!json_file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Write the JSON preamble
        json_file << "[\n";

        bool first_function = true;

        for (const auto &[name, stats] : function_stats_) {
            for (const auto &call : stats.calls) {
                if (!first_function) {
                    json_file << ",\n";
                }
                first_function = false;

                json_file << "  {\n";
                json_file << "    \"name\": \"" << name << "\",\n";
                json_file
                    << "    \"ph\": \"X\",\n"; // "X" indicates a complete event
                json_file << "    \"ts\": " << call.start_time << ",\n";
                json_file << "    \"dur\": "
                          << (call.end_time - call.start_time) << ",\n";
                json_file << "    \"pid\": 0,\n"; // Process ID (arbitrary, use
                                                  // 0 for simplicity)
                json_file << "    \"tid\": 0\n";  // Thread ID (arbitrary, use 0
                                                  // for simplicity)
                json_file << "  }";
            }
        }

        // End the JSON array
        json_file << "\n]\n";

        json_file.close();
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

    std::unordered_map<std::string, function_stats> function_stats_;
};

// auto_profiler, start timing and end timing
class auto_profiler {
  public:
    auto_profiler(const std::string &function_name)
        : function_name_(function_name),
          start_time_(ntt_profiler::get_instance().start_timing()) {}

    ~auto_profiler() {
        ntt_profiler::get_instance().end_timing(function_name_, start_time_);
    }

  private:
    std::string function_name_;
    uint64_t start_time_;
};

// #define AUTO_NTT_PROFILER auto_profiler profiler(__FUNCTION__);
// #define DISP_NTT_PROFILER ntt_profiler::get_instance().console_print();
} // namespace nncase::ntt