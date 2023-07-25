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
#include "dump_manager.h"
#include "model.h"
#include "result.h"
#include "runtime_module.h"
#include "runtime_tensor.h"
#include <gsl/gsl-lite.hpp>
#include <istream>
#include <memory>
#include <nncase/shape.h>
#include <nncase/tensor.h>
#include <nncase/type.h>
#include <unordered_map>
#include <variant>

BEGIN_NS_NNCASE_RUNTIME

class NNCASE_API options_dict {
  public:
    template <class T> result<T> get_scalar_opt(const char *name) {
        try_var(value, get<scalar>(name));
        return value.template as<T>();
    }

    template <class T> result<T> get(const char *name) {
        auto it = values_.find(name);
        if (it != values_.end())
            return ok(std::get<T>(it->second));
        else
            return err(std::errc::result_out_of_range);
    }

    template <class T> result<void> set(const char *name, T value) {
        values_[name] = value;
        return ok();
    }

  private:
    std::unordered_map<const char *, std::variant<scalar, std::string>> values_;
};

struct tensor_desc {
    typecode_t datatype;
    size_t start;
    size_t size;
};

class NNCASE_API interpreter {
  public:
    interpreter() noexcept;
    interpreter(interpreter &) = delete;
    interpreter(interpreter &&) = default;

    [[nodiscard]] result<void> load_model(gsl::span<const gsl::byte> buffer,
                                          bool copy_buffer = false) noexcept;

    [[nodiscard]] result<void> load_model(std::istream &stream) noexcept;

    options_dict &options() noexcept;
    result<runtime_module *> find_module_by_id(size_t index) noexcept;
    result<size_t> find_id_by_module(runtime_module *module) noexcept;

    /* V1 APIs */

    size_t inputs_size() const noexcept;
    size_t outputs_size() const noexcept;
    tensor_desc input_desc(size_t index) const noexcept;
    tensor_desc output_desc(size_t index) const noexcept;
    dims_t input_shape(size_t index) const noexcept;
    dims_t output_shape(size_t index) const noexcept;
    result<runtime_tensor> input_tensor(size_t index) noexcept;
    result<void> input_tensor(size_t index, runtime_tensor tensor) noexcept;
    result<runtime_tensor> output_tensor(size_t index) noexcept;
    result<void> output_tensor(size_t index, runtime_tensor tensor) noexcept;

    result<void> run() noexcept;

    /* V2 APIs */

    result<runtime_function *>
    find_function_by_name(std::string_view name) noexcept;
    result<runtime_function *> entry_function() noexcept;
    std::shared_ptr<nncase::runtime::dump_manager> dump_manager() noexcept {
        if (!dump_manager_) {
            dump_manager_ = std::make_shared<nncase::runtime::dump_manager>();
        }
        return dump_manager_;
    }

  private:
    tensor_type input_tensor_type(size_t index) const noexcept;
    tensor_type output_tensor_type(size_t index) const noexcept;

    result<void> initialize_model(const model_header &header) noexcept;

  private:
    std::shared_ptr<nncase::runtime::dump_manager> dump_manager_;
    std::vector<std::unique_ptr<runtime_module>> modules_;
    runtime_function *entry_function_;
    options_dict options_;
    std::vector<runtime_tensor> input_tensors_;
    std::vector<runtime_tensor> output_tensors_;
};

END_NS_NNCASE_RUNTIME
