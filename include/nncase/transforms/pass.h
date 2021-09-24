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
#include <filesystem>
#include <nncase/ir/function.h>
//#include <nncase/ir/quantizer.h>
#include <optional>
#include <vector>

namespace nncase {
class target;

namespace schedule {
class function_schedule_context;
}
} // namespace nncase

namespace nncase::ir::transforms {
struct run_pass_options {
    nncase::target *target;
    // ir::quantizer *quantizer;
    schedule::function_schedule_context *schedule_context;
    std::optional<std::filesystem::path> dump_dir;
};

/** @brief Function level pass */
class NNCASE_API function_pass {
  public:
    function_pass(std::string name = "") : name_(name_) {}
    virtual ~function_pass() = default;
    function_pass(const function_pass &) = delete;
    function_pass(function_pass &&) = default;
    function_pass &operator=(const function_pass &) = delete;

    /** @brief Get the name of the pass */
    const std::string &name() const noexcept { return name_; }

    /** @brief Run the pass on the function */
    void run(const function &func, const run_pass_options &options);

  protected:
    virtual void run_core(const function &func,
                          const run_pass_options &options) = 0;

  private:
    std::string name_;
};

class NNCASE_API pass_manager {
  public:
    pass_manager(const function &func, run_pass_options options)
        : func_(func), options_(std::move(options)) {}
    pass_manager(const pass_manager &) = delete;
    pass_manager &operator=(const pass_manager &) = delete;

    function_pass &emplace(std::unique_ptr<function_pass> pass);

    void run();

  private:
    std::vector<std::unique_ptr<function_pass>> passes_;
    function func_;
    run_pass_options options_;
};
} // namespace nncase::ir::transforms
