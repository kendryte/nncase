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
#include "allocator.h"
#include "model.h"
#include "result.h"
#include "runtime_module.h"
#include <gsl/gsl-lite.hpp>
#include <memory>
#include <unordered_map>

BEGIN_NS_NNCASE_RUNTIME

class NNCASE_API options_dict
{
public:
    template <class T>
    result<T> get(const char *name)
    {
        auto it = values_.find(name);
        if (it != values_.end())
            return ok(it->second.as<T>());
        else
            return err(std::errc::result_out_of_range);
    }

    template <class T>
    result<void> set(const char *name, T value)
    {
        values_[name] = scalar(value);
        return ok();
    }

private:
    std::unordered_map<const char *, scalar> values_;
};

class NNCASE_API interpreter
{
public:
    interpreter() noexcept;
    interpreter(interpreter &) = delete;
    interpreter(interpreter &&) = default;

    NNCASE_NODISCARD result<void> load_model(gsl::span<const gsl::byte> buffer) noexcept;

    size_t inputs_size() const noexcept;
    size_t outputs_size() const noexcept;
    const memory_range &input_desc(size_t index) const noexcept;
    const memory_range &output_desc(size_t index) const noexcept;
    const runtime_shape_t &input_shape(size_t index) const noexcept;
    const runtime_shape_t &output_shape(size_t index) const noexcept;
    result<runtime_tensor> input_tensor(size_t index) noexcept;
    result<void> input_tensor(size_t index, runtime_tensor tensor) noexcept;
    result<runtime_tensor> output_tensor(size_t index) noexcept;
    result<void> output_tensor(size_t index, runtime_tensor tensor) noexcept;

    result<void> run() noexcept;

    result<runtime_module *> find_module_by_id(size_t index) noexcept;
    options_dict &options() noexcept;

private:
    std::vector<std::unique_ptr<runtime_module>> modules_;
    runtime_module *main_module_;
    options_dict options_;
};

END_NS_NNCASE_RUNTIME
