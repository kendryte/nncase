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
#include <nncase/runtime/runtime_module.h>
#include <nncase/runtime/span_reader.h>

using namespace nncase;
using namespace nncase::runtime;

namespace
{
struct module_pointers
{
    gsl::span<const mempool_desc> mempools;

    module_pointers(const module_header &header) noexcept
    {
        span_reader reader(gsl::make_span(reinterpret_cast<const gsl::byte *>(&header), header.size));
        reader.skip(sizeof(module_header));
        reader.read_span(mempools, header.mempools);
    }
};
}

runtime_module::runtime_module(const module_header &header) noexcept
    : header_(header)
{
}

const module_type_t &runtime_module::type() const noexcept
{
    return header_.type;
}

uint32_t runtime_module::mempools_count() const noexcept
{
    return header_.mempools;
}

const mempool_desc &runtime_module::mempool_desc(size_t index) const noexcept
{
    return module_pointers(header_).mempools[index];
}

uint32_t runtime_module::inputs_count() const noexcept
{
    return header_.inputs;
}

uint32_t runtime_module::outputs_count() const noexcept
{
    return header_.outputs;
}

//const shape_header &runtime_module::input_shape(size_t index) const noexcept
//{
//}

result<void> runtime_module::initialize(interpreter &interp) noexcept
{
    return ok();
}
