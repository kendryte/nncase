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
#include <cassert>
#include <iostream>
#include <nncase/runtime/error.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/span_reader.h>

using namespace nncase;
using namespace nncase::runtime;

interpreter::interpreter() noexcept
    : main_module_(nullptr)
{
}

result<void> interpreter::load_model(gsl::span<const gsl::byte> buffer) noexcept
{
    span_reader reader(buffer);
    auto header = reader.get_ref<model_header>();
    // 1. Validate model
    if (header->identifier != MODEL_IDENTIFIER)
        return err(nncase_errc::invalid_model_indentifier);
    if (header->version != MODEL_VERSION)
        return err(nncase_errc::invalid_model_version);

    // 2. Load modules
    try
    {
        modules_.resize(header->modules);
    }
    catch (...)
    {
        return err(std::errc::not_enough_memory);
    }

    for (size_t i = 0; i < header->modules; i++)
    {
        auto mod_header = reader.get_ref<module_header>();
        reader.skip(mod_header->size);
        try_var(rt_module, runtime_module::create(mod_header->type));

        try_(rt_module->initialize(*mod_header, *this));
        if (i == header->main_module)
            main_module_ = rt_module.get();
        modules_[i] = std::move(rt_module);
    }

    for (auto &mod : modules_)
        try_(mod->initialize_inter_modules(*this));

    return ok();
}

size_t interpreter::inputs_size() const noexcept
{
    return main_module_->inputs_size();
}

size_t interpreter::outputs_size() const noexcept
{
    return main_module_->outputs_size();
}

const memory_range &interpreter::input_desc(size_t index) const noexcept
{
    return main_module_->input_desc(index);
}

const memory_range &interpreter::output_desc(size_t index) const noexcept
{
    return main_module_->output_desc(index);
}

const runtime_shape_t &interpreter::input_shape(size_t index) const noexcept
{
    return main_module_->input_shape(index);
}

const runtime_shape_t &interpreter::output_shape(size_t index) const noexcept
{
    return main_module_->output_shape(index);
}

result<runtime_tensor> interpreter::input_tensor(size_t index) noexcept
{
    return main_module_->input_tensor(index);
}

result<void> interpreter::input_tensor(size_t index, runtime_tensor tensor) noexcept
{
    return main_module_->input_tensor(index, tensor);
}

result<runtime_tensor> interpreter::output_tensor(size_t index) noexcept
{
    return main_module_->output_tensor(index);
}

result<void> interpreter::output_tensor(size_t index, runtime_tensor tensor) noexcept
{
    return main_module_->output_tensor(index, tensor);
}

result<void> interpreter::run() noexcept
{
    return main_module_->run();
}

result<runtime_module *> interpreter::find_module_by_id(size_t index) noexcept
{
    if (index < modules_.size())
        return ok(modules_[index].get());
    return err(std::errc::result_out_of_range);
}

options_dict &interpreter::options() noexcept
{
    return options_;
}
