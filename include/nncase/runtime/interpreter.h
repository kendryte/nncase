/* Copyright 2019-2020 Canaan Inc.
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
#include <gsl/gsl-lite.hpp>
#include <memory>

BEGIN_NS_NNCASE_RUNTIME

class NNCASE_API interpreter
{
public:
    interpreter(allocator &allocator, allocation_state &alloc_state);

    NNCASE_NODISCARD std::error_condition load_model(gsl::span<const uint8_t> buffer) noexcept;

    size_t inputs_size() const noexcept { return model_header_->inputs; }
    size_t outputs_size() const noexcept { return model_header_->outputs; }

    const runtime_shape_t &input_shape_at(size_t index) const noexcept { return input_shapes_.at(index); }
    const memory_range &input_at(size_t index) const noexcept { return inputs_[index]; }
    const memory_range &output_at(size_t index) const noexcept { return outputs_[index]; }

    template <class T>
    gsl::span<T> memory_at(const memory_range &range) const noexcept
    {
        auto span = memory_at(range);
        return { reinterpret_cast<T *>(span.data()), span.size() / sizeof(T) };
    }

    std::error_condition run();

protected:
    NNCASE_NODISCARD virtual std::error_condition initialize();
    virtual gsl::span<uint8_t> memory_at(const memory_range &range) const noexcept;

private:
    void step();

private:
    allocator &allocator_;
    allocation_state &alloc_state_;
    const model_header *model_header_;
    std::unique_ptr<uint8_t[]> main_mem_;
    gsl::span<const memory_range> inputs_;
    gsl::span<const memory_range> outputs_;
    gsl::span<const runtime_shape_t> input_shapes_;
    gsl::span<const runtime_shape_t> output_shapes_;
};

END_NS_NNCASE_RUNTIME
