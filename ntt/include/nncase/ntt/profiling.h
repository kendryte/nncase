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
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nncase::ntt::runtime {

enum class profiling_level { kernel, device };

static std::string to_string(profiling_level level) {
    switch (level) {
    case profiling_level::kernel:
        return "kernel";
    case profiling_level::device:
        return "device";
    default:
        return "unknown";
    }
}

template <class TOPOLOGY> class timer_record_base {
  public:
    struct call_instance {
        uint64_t start_time;
        uint64_t end_time;
    };

    struct function_stats {
        uint64_t call_count = 0;
        uint64_t total_time = 0;
        profiling_level level;
        std::vector<call_instance> calls;
    };

    // 判断记录是否有效
    virtual bool is_valid() const = 0;

    // 设置计时记录
    virtual void set_time(std::string_view function_name, uint64_t start_time,
                          uint64_t end_time) = 0;

    // 控制台打印统计信息
    virtual void console_print() const = 0;

    // 导出为 CSV 文件
    virtual void csv_print(std::string_view filename) const = 0;

    // 导出为 JSON 文件
    virtual void markdown_print(std::string_view filename) const = 0;

    // 导出为 JSON 文件
    virtual void json_print(std::string_view filename) const = 0;

    // 设置记录 ID
    virtual void set_id(TOPOLOGY id) = 0;

    virtual void set_level(std::string_view filename,
                           profiling_level level) = 0;

    // 虚析构函数，确保子类正确释放资源
    virtual ~timer_record_base() = default;

  protected:
    timer_record_base() = default; // 禁止直接实例化抽象类
    TOPOLOGY instance_id_;
    std::unordered_map<std::string_view, function_stats> function_stats_;
};
} // namespace nncase::ntt::runtime
