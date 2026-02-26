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
#include <nncase/runtime/char_array_buffer.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/span_reader.h>
#include <nncase/runtime/stream_reader.h>

using namespace nncase;
using namespace nncase::runtime;

interpreter::interpreter() noexcept : entry_function_(nullptr) {
    options().set("profiling", (uint8_t)0);
}

result<void> interpreter::load_model(gsl::span<const gsl::byte> buffer,
                                     bool copy_buffer) noexcept {
    if (copy_buffer) {
        char_array_buffer array_buffer(buffer.as_span<const char>());
        std::istream stream(&array_buffer);
        return load_model(stream);
    }

    span_reader reader(buffer);
    auto &header = *reader.get_ref<model_header>();
    try_(initialize_model(header));

    for (size_t i = 0; i < header.modules; i++) {
        auto mod_type = reader.peek_with_offset<decltype(module_header::kind)>(
            offsetof(module_header, kind));
        auto mod_size = reader.peek_with_offset<decltype(module_header::size)>(
            offsetof(module_header, size));
        auto payload = reader.read_span(mod_size);
        try_var(rt_module, runtime_module::create(mod_type));

        try_(rt_module->initialize(payload, *this));
        if (header.entry_module != MODEL_HAS_NO_ENTRY) {
            if (i == header.entry_module) {
                try_set(entry_function_,
                        rt_module->find_function_by_id(header.entry_function));
            }
        }

        modules_[i] = std::move(rt_module);
    }

    return ok();
}

result<void> interpreter::load_model(std::istream &stream) noexcept {
    stream_reader reader(stream);
    auto header = reader.read<model_header>();
    try_(initialize_model(header));

    std::streampos module_pos = reader.tell();
    for (size_t i = 0; i < header.modules; i++) {
        reader.seek(module_pos);

        auto mod_header = reader.read<module_header>();
        try_var(rt_module, runtime_module::create(mod_header.kind));

        reader.seek(module_pos);
        try_(rt_module->initialize(reader, *this));
        if (header.entry_module != MODEL_HAS_NO_ENTRY) {
            if (i == header.entry_module) {
                try_set(entry_function_,
                        rt_module->find_function_by_id(header.entry_function));
            }
        }

        modules_[i] = std::move(rt_module);
        module_pos += mod_header.size;
    }

    return ok();
}

result<void>
interpreter::initialize_model(const model_header &header) noexcept {
    entry_function_ = nullptr;
    // 1. Validate model
    if (header.identifier != MODEL_IDENTIFIER)
        return err(nncase_errc::invalid_model_indentifier);
    if (header.version != MODEL_VERSION)
        return err(nncase_errc::invalid_model_version);

    // 2. Load modules
    try {
        modules_.resize(header.modules);
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    return ok();
}

size_t interpreter::inputs_size() const noexcept {
    return entry_function_->parameters_size();
}

size_t interpreter::outputs_size() const noexcept {
    auto &ret_type = entry_function_->return_type();
    auto tuple_t = ret_type.as<tuple_type>();
    return tuple_t.is_ok() ? tuple_t.unwrap()->fields().size() : 1;
}

tensor_type interpreter::input_tensor_type(size_t index) const noexcept {
    return entry_function_->parameter_type(index)
        .expect("Invalid input index")
        .as<tensor_type>()
        .expect("Not a tensor type");
}
tensor_type interpreter::output_tensor_type(size_t index) const noexcept {
    auto &ret_type = entry_function_->return_type();
    auto tuple_t = ret_type.as<tuple_type>();
    return (tuple_t.is_ok() ? tuple_t.unwrap()->fields()[index] : ret_type)
        .as<tensor_type>()
        .expect("Not a tensor type");
}

tensor_desc interpreter::input_desc(size_t index) const noexcept {
    auto type = input_tensor_type(index)->dtype();
    auto dtype = type.as<prim_type_t>().expect("Not a prim type");
    auto size_bytes = get_bytes(dtype, input_shape(index));
    size_t start = 0;
    for (size_t i = 0; i < index; i++) {
        auto type = input_tensor_type(i)->dtype();
        auto dtype = type.as<prim_type_t>().expect("Not a prim type");
        start += get_bytes(dtype, input_shape(i));
    }
    return {dtype->typecode(), start, size_bytes};
}

tensor_desc interpreter::output_desc(size_t index) const noexcept {
    auto type = output_tensor_type(index)->dtype();
    auto dtype = type.as<prim_type_t>().expect("Not a prim type");
    auto size_bytes = get_bytes(dtype, output_shape(index));
    size_t start = 0;
    for (size_t i = 0; i < index; i++) {
        auto type = output_tensor_type(i)->dtype();
        auto dtype = type.as<prim_type_t>().expect("Not a prim type");
        start += get_bytes(dtype, output_shape(i));
    }
    return {dtype->typecode(), start, size_bytes};
}

dims_t interpreter::input_shape(size_t index) const noexcept {
    auto type = input_tensor_type(index);
    return type->shape().as_fixed().expect("Not fixed shape");
}

dims_t interpreter::output_shape(size_t index) const noexcept {
    auto type = output_tensor_type(index);
    return type->shape().as_fixed().expect("Not fixed shape");
}

result<runtime_tensor> interpreter::input_tensor(size_t index) noexcept {
    CHECK_WITH_ERR(index < inputs_size(), std::errc::result_out_of_range);
    if (input_tensors_.empty()) {
        input_tensors_.resize(inputs_size());
    }
    return ok(input_tensors_[index]);
}

result<void> interpreter::input_tensor(size_t index,
                                       runtime_tensor tensor) noexcept {
    CHECK_WITH_ERR(index < inputs_size(), std::errc::result_out_of_range);
    if (input_tensors_.empty()) {
        input_tensors_.resize(inputs_size());
    }
    input_tensors_[index] = tensor;
    return ok();
}

result<runtime_tensor> interpreter::output_tensor(size_t index) noexcept {
    CHECK_WITH_ERR(index < outputs_size(), std::errc::result_out_of_range);
    if (output_tensors_.empty()) {
        output_tensors_.resize(outputs_size());
    }
    return ok(output_tensors_[index]);
}

result<void> interpreter::output_tensor(size_t index,
                                        runtime_tensor tensor) noexcept {
    CHECK_WITH_ERR(index < outputs_size(), std::errc::result_out_of_range);
    if (output_tensors_.empty()) {
        output_tensors_.resize(outputs_size());
    }
    output_tensors_[index] = tensor;
    return ok();
}

void interpreter::set_profiling(uint8_t enabled) noexcept {
    options().set("profiling", enabled);
}

result<void> interpreter::run() noexcept {
    std::vector<value_t> params(inputs_size(), nullptr);
    for (size_t i = 0; i < params.size(); i++) {
        try_var(in, input_tensor(i));
        params[i] = in.impl();
    }

    auto is_tensor_output = entry_function_->return_type().is_a<tensor_type>();
    if (output_tensors_.empty()) {
        try_var(ret_value, entry_function_->invoke(params));
        if (is_tensor_output) {
            try_var(t, ret_value.as<tensor>());
            try_(output_tensor(0, runtime_tensor(t)));
        } else {
            try_var(tp, ret_value.as<tuple>());
            for (size_t i = 0; i < tp->fields().size(); i++) {
                try_var(t, tp->fields()[i].as<tensor>());
                try_(output_tensor(i, runtime_tensor(t)));
            }
        }
    } else {
        std::vector<value_t> ret_fields(outputs_size(), nullptr);
        for (size_t i = 0; i < ret_fields.size(); i++) {
            try_var(out, output_tensor(i));
            ret_fields[i] = out.impl();
        }

        try_(entry_function_->invoke(
            params, is_tensor_output
                        ? ret_fields[0]
                        : tuple(std::in_place, std::move(ret_fields))));
    }

    return ok();
}

result<runtime_module *> interpreter::find_module_by_id(size_t index) noexcept {
    CHECK_WITH_ERR(index < modules_.size(), std::errc::result_out_of_range);
    return ok(modules_[index].get());
}

result<size_t> interpreter::find_id_by_module(runtime_module *module) noexcept {
    auto it = std::find_if(modules_.begin(), modules_.end(),
                           [&module](const std::unique_ptr<runtime_module> &p) {
                               return p.get() == module;
                           });
    if (it == modules_.end()) {
        return err(std::errc::result_out_of_range);
    }
    return ok((it - modules_.begin()));
}

options_dict &interpreter::options() noexcept { return options_; }

result<runtime_function *> interpreter::entry_function() noexcept {
    if (entry_function_)
        return ok(entry_function_);
    return err(std::errc::no_such_file_or_directory);
}
