/* Copyright 2020 Canaan Inc.
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
#include "transform.h"
#include <filesystem>
#include <optional>
#include <vector>

namespace nncase
{
class target;
}

namespace nncase::ir::transforms
{
class NNCASE_API pass
{
public:
    pass(std::string dump_name = "")
        : dump_name_(dump_name) { }
    pass(pass &) = delete;
    pass(pass &&) = default;

    pass &operator=(pass &) = delete;

    void run(graph &graph, nncase::target &target, ir::quantizer *quantizer, std::optional<std::filesystem::path> dump_dir);

    template <class T, class... TArgs>
    transform *emplace(TArgs &&... args)
    {
        return static_cast<T *>(transforms_.emplace_back(new T(std::forward<TArgs>(args)...)).get());
    }

    const std::string &name() const noexcept { return dump_name_; }

private:
    std::vector<std::unique_ptr<transform>> transforms_;
    std::string dump_name_;
};

class NNCASE_API pass_manager
{
public:
    pass_manager(graph &graph, nncase::target &target)
        : graph_(graph), target_(target), quantizer_(nullptr) { }
    pass_manager(pass_manager &) = delete;

    void add_pass(pass &&pass);
    void run();

    void dump_dir(const std::filesystem::path &dir);
    void quantizer(ir::quantizer *q);

private:
    std::vector<pass> passes_;
    graph &graph_;
    nncase::target &target_;
    ir::quantizer *quantizer_;
    std::optional<std::filesystem::path> dump_dir_;
};
}
