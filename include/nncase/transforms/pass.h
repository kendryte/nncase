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
struct run_pass_options
{
    ir::quantizer *quantizer;
    schedule::function_schedule_context *schedule_context;
    std::optional<std::filesystem::path> dump_dir;
};

class NNCASE_API pass
{
public:
    pass(std::string dump_name = "")
        : dump_name_(dump_name) { }
    virtual ~pass() = default;
    pass(pass &) = delete;
    pass(pass &&) = default;

    pass &operator=(pass &) = delete;

    void run(graph &graph, nncase::target &target, const run_pass_options &options);

    const std::string &name() const noexcept { return dump_name_; }

protected:
    virtual void run_core(graph &graph, nncase::target &target, const run_pass_options &options) = 0;

private:
    std::string dump_name_;
};

class NNCASE_API transform_pass : public pass
{
public:
    using pass::pass;

    transform_pass(transform_pass &) = delete;
    transform_pass(transform_pass &&) = default;

    transform_pass &operator=(transform_pass &) = delete;

    template <class T, class... TArgs>
    transform *emplace(TArgs &&...args)
    {
        return static_cast<T *>(transforms_.emplace_back(new T(std::forward<TArgs>(args)...)).get());
    }

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;

private:
    std::vector<std::unique_ptr<transform>> transforms_;
};

class NNCASE_API graph_pass : public pass
{
public:
    using pass::pass;

    graph_pass(graph_pass &) = delete;
    graph_pass(graph_pass &&) = default;

    graph_pass &operator=(graph_pass &) = delete;
};

class NNCASE_API pass_manager
{
public:
    pass_manager(graph &graph, nncase::target &target)
        : graph_(graph), target_(target), quantizer_(nullptr), schedule_context_(nullptr) { }
    pass_manager(pass_manager &) = delete;

    template <class TPass = transform_pass, class... TArgs>
    void add_pass(TArgs &&...pass)
    {
        passes_.emplace_back(std::make_unique<TPass>(std::forward<TArgs>(pass)...));
    }

    void run();

    void dump_dir(const std::filesystem::path &dir);
    void quantizer(ir::quantizer *q);
    void schedule_context(schedule::function_schedule_context *c);

private:
    std::vector<std::unique_ptr<pass>> passes_;
    graph &graph_;
    nncase::target &target_;
    ir::quantizer *quantizer_;
    schedule::function_schedule_context *schedule_context_;
    std::optional<std::filesystem::path> dump_dir_;
};
}
