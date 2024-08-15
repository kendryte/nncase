#pragma once
#include <chrono>
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
        for (const auto &[name, stats] : functionStats_) {
            std::cout << "Function: " << name << "\n";
            std::cout << "  Calls: " << stats.callCount << "\n";
            std::cout << "  Total time: " << stats.totalTime
                      << " microseconds\n";
        }
    }

  private:
    FunctionProfiler() = default;

    ~FunctionProfiler() { printStatistics(); }

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

#ifdef NTT_PROFILER
#define AUTO_NTT_PROFILER AutoProfiler profiler(__FUNCTION__);
#define DISP_NTT_PROFILER FunctionProfiler::getInstance().printStatistics();
#else
#define AUTO_NTT_PROFILER
#define DISP_NTT_PROFILER
#endif
