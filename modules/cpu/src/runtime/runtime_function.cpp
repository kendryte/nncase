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
#include "runtime_function.h"
#include "elfloader.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;

namespace {
typedef struct memory_range {
    uint32_t start;
    uint32_t size;
} memory_range_t;

typedef struct desc_header {
    uint32_t input_pool_size;

    uint32_t output_pool_size;

    uint32_t inputs;

    uint32_t outputs;
} desc_header_t;

} // namespace

cpu_runtime_module &cpu_runtime_function::module() const noexcept {
    return static_cast<cpu_runtime_module &>(runtime_function::module());
}

result<void> cpu_runtime_function::initialize_core(
    NNCASE_UNUSED runtime_function_init_context &context) noexcept {

    try_(context.read_section(".desc", [this](auto sr, size_t) -> result<void> {
        auto header = sr.template read<desc_header>();
        if (parameters_size() != header.inputs + header.outputs)
            return nncase::err(std::errc::invalid_argument);

        for (uint32_t i = 0; i < header.inputs; i++) {
            sr.template read<memory_range>();
            auto rank = sr.template read<uint32_t>();
            std::vector<uint32_t> shape(rank);
            std::cout << "shape: ";
            for (uint32_t j = 0; j < rank; j++) {
                shape[j] = sr.template read<uint32_t>();
                std::cout << shape[j] << ", ";
            }
            std::cout << std::endl;

            std::vector<uint32_t> stride(rank);
            std::cout << "stride: ";
            for (uint32_t j = 0; j < rank; j++) {
                stride[j] = sr.template read<uint32_t>();
                std::cout << stride[j] << ", ";
            }
            std::cout << std::endl;

            input_ranks_.emplace_back(rank);
            input_shapes_.emplace_back(shape);
            input_strides_.emplace_back(stride);
        }

        for (uint32_t i = 0; i < header.outputs; i++) {
            sr.template read<memory_range>();
            auto rank = sr.template read<uint32_t>();
            std::vector<uint32_t> shape(rank);
            std::cout << "shape: ";
            for (uint32_t j = 0; j < rank; j++) {
                shape[j] = sr.template read<uint32_t>();
                std::cout << shape[j] << ", ";
            }
            std::cout << std::endl;

            std::vector<uint32_t> stride(rank);
            std::cout << "stride: ";
            for (uint32_t j = 0; j < rank; j++) {
                stride[j] = sr.template read<uint32_t>();
                std::cout << stride[j] << ", ";
            }
            std::cout << std::endl;

            output_ranks_.emplace_back(rank);
            output_shapes_.emplace_back(shape);
            output_strides_.emplace_back(stride);
        }

        return ok();
    }));

    return ok();
}

result<value_t>
cpu_runtime_function::invoke_core(NNCASE_UNUSED gsl::span<value_t> parameters,
                                  NNCASE_UNUSED value_t return_value) noexcept {
    try_var(id, module().find_id_by_function(this));
    std::cout << "call " << id << std::endl;

    std::vector<buffer_t *> buffers(input_ranks_.size() + output_ranks_.size());

    // input buffer
    for (uint32_t i = 0; i < input_ranks_.size(); i++) {
        auto input_tensor = parameters[i].as<tensor>().expect(
            "input " + std::to_string(i) + " is not a tensor");
        try_var(input_span, get_input_span(input_tensor));
        buffer_t *input_buffer =
            new buffer_t{input_span.data(), 0, input_shapes_[i].data(),
                         input_strides_[i].data(), input_ranks_[i]};
        buffers[i] = input_buffer;
    }

    // output buffer
    for (uint32_t i = 0; i < output_ranks_.size(); i++) {
        auto output_tensor =
            parameters[input_ranks_.size() + i].as<tensor>().expect(
                "output " + std::to_string(i) + " is not a tensor");
        try_var(output_span, get_output_span(output_tensor));
        buffer_t *output_buffer =
            new buffer_t{output_span.data(), 0, output_shapes_[i].data(),
                         output_strides_[i].data(), output_ranks_[i]};
        buffers[input_ranks_.size() + i] = output_buffer;
    }

    auto elfloader_ = elfloader{(char *)module().text_physical().data()};
    elfloader_.invoke_elf(id, buffers.data(), &nncase_mt, nullptr, (void *)module().rdata_physical().data());
    for (int i = 0; i < buffers.size(); i++) {
        delete buffers[i];
    }

    return ok<value_t>(tuple(std::in_place));
}