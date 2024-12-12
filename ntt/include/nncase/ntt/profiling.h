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

namespace nncase::ntt {
class ntt_profiler {
  public:
    struct function_stats {
        uint64_t call_count = 0;
        uint64_t total_time = 0;
    };

    static ntt_profiler &get_instance() {
        static ntt_profiler instance;
        return instance;
    }

    // record start time
    uint64_t start_timing() { return get_current_time(); }

    // record end time and calculate duration
    void end_timing(const std::string &function_name, uint64_t start_time) {
        uint64_t end_time = get_current_time();
        uint64_t duration = end_time - start_time;

        auto &stats = function_stats_[function_name];
        stats.call_count++;
        stats.total_time += duration;
    }

    // print statistics
    void print_statistics() const {
        uint64_t total_time = 0;
        for (const auto &[name, stats] : function_stats_) {
            total_time += stats.total_time;
        }

        std::cout << "\033[34m\nStatistics for NTT kernels. Total time: "
                  << total_time
                  << " microseconds. More info in: ./ntt_profiler.md\033[0m\n";
        for (const auto &[name, stats] : function_stats_) {
            std::cout << "Function: " << name << "\n";
            std::cout << "  Calls: " << stats.call_count << "\n";
            std::cout << "  Total time: " << stats.total_time
                      << " microseconds\n";
            std::cout << "  Time Ratio: " << std::fixed << std::setprecision(2)
                      << static_cast<double>(stats.total_time) /
                             static_cast<double>(total_time)
                      << std::endl;
        }
    }

    void write_markdown_report(const std::string &filename) const {

        uint64_t total_time = 0;
        for (const auto &[name, stats] : function_stats_) {
            total_time += stats.total_time;
        }

        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        ofs << "# Statistics for NTT Kernels\n\n";
        ofs << "**Total time:** `" << total_time << "` microseconds\n\n";
        ofs << "| Function Name | Calls | Total Time (microseconds) | Time "
               "Ratio |\n";
        ofs << "|---------------|-------|--------------------------|-----------"
               "-|\n";

        for (const auto &[name, stats] : function_stats_) {
            ofs << "| " << name << " | " << stats.call_count << " | "
                << stats.total_time << " | " << std::fixed
                << std::setprecision(2)
                << static_cast<double>(stats.total_time) /
                       static_cast<double>(total_time)
                << " |\n";
        }

        ofs << "\n*Note*: The `Time Ratio` is the fraction of the total time "
               "taken by each function.\n";
    }

  private:
    ntt_profiler() = default;

    ~ntt_profiler() {
        print_statistics();
        write_markdown_report("ntt_profiler.md");
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
// #define DISP_NTT_PROFILER ntt_profiler::get_instance().print_statistics();
} // namespace nncase::ntt