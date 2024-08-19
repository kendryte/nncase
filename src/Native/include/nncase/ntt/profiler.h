#pragma once
#ifdef NTT_PROFILER
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

class FunctionProfiler {
  public:
    struct FunctionStats {
        uint64_t callCount = 0;
        uint64_t totalTime = 0;
    };

    static FunctionProfiler &getInstance() {
        static FunctionProfiler instance;
        return instance;
    }

    // record start time
    uint64_t startTiming() { return getCurrentTime(); }

    // record end time and calculate duration
    void endTiming(const std::string &functionName, uint64_t startTime) {
        uint64_t endTime = getCurrentTime();
        uint64_t duration = endTime - startTime;

        auto &stats = functionStats_[functionName];
        stats.callCount++;
        stats.totalTime += duration;
    }

    // print statistics
    void printStatistics() const {
        uint64_t totalTime = 0;
        for (const auto &[name, stats] : functionStats_) {
            totalTime += stats.totalTime;
        }

        std::cout << "\033[34m\nStatistics for NTT kernels. Total time: "
                  << totalTime
                  << " microseconds. More info in: ./ntt_profiler.md\033[0m\n";
        for (const auto &[name, stats] : functionStats_) {
            std::cout << "Function: " << name << "\n";
            std::cout << "  Calls: " << stats.callCount << "\n";
            std::cout << "  Total time: " << stats.totalTime
                      << " microseconds\n";
            std::cout << "  Time Ratio: " << std::fixed << std::setprecision(2)
                      << static_cast<double>(stats.totalTime) /
                             static_cast<double>(totalTime)
                      << std::endl;
        }
    }

    void writeMarkdownReport(const std::string &filename) const {

        uint64_t totalTime = 0;
        for (const auto &[name, stats] : functionStats_) {
            totalTime += stats.totalTime;
        }

        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        ofs << "# Statistics for NTT Kernels\n\n";
        ofs << "**Total time:** `" << totalTime << "` microseconds\n\n";
        ofs << "| Function Name | Calls | Total Time (microseconds) | Time "
               "Ratio |\n";
        ofs << "|---------------|-------|--------------------------|-----------"
               "-|\n";

        for (const auto &[name, stats] : functionStats_) {
            ofs << "| " << name << " | " << stats.callCount << " | "
                << stats.totalTime << " | " << std::fixed
                << std::setprecision(2)
                << static_cast<double>(stats.totalTime) /
                       static_cast<double>(totalTime)
                << " |\n";
        }

        ofs << "\n*Note*: The `Time Ratio` is the fraction of the total time "
               "taken by each function.\n";
    }

  private:
    FunctionProfiler() = default;

    ~FunctionProfiler() {
        printStatistics();
        writeMarkdownReport("ntt_profiler.md");
    }

    uint64_t getCurrentTime() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    }

    std::unordered_map<std::string, FunctionStats> functionStats_;
};

// AutoProfiler, start timing and end timing
class AutoProfiler {
  public:
    AutoProfiler(const std::string &functionName)
        : functionName_(functionName),
          startTime_(FunctionProfiler::getInstance().startTiming()) {}

    ~AutoProfiler() {
        FunctionProfiler::getInstance().endTiming(functionName_, startTime_);
    }

  private:
    std::string functionName_;
    uint64_t startTime_;
};

#define AUTO_NTT_PROFILER AutoProfiler profiler(__FUNCTION__);
#define DISP_NTT_PROFILER FunctionProfiler::getInstance().printStatistics();

#else // 如果NTT_PROFILER未定义

class AutoProfiler {
  public:
    AutoProfiler() {}
};

#define AUTO_NTT_PROFILER
#define DISP_NTT_PROFILER

#endif
