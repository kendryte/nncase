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
#include "runtime_module.h"
#include "runtime_function.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

gsl::span<gsl::byte> stackvm_runtime_module::data() const noexcept
{
    if (!data_.empty())
    {
        auto &block = static_cast<const detail::host_runtime_tensor_impl *>(data_tensor().impl())->memory_block();
        return block.virtual_buffer();
    }

    return {};
}

gsl::span<const gsl::byte> stackvm_runtime_module::rdata() const noexcept
{
    return rdata_;
}

const runtime_tensor &stackvm_runtime_module::data_tensor() const noexcept
{
    return data_;
}

result<void> stackvm_runtime_module::initialize_before_functions(runtime_module_init_context &context) noexcept
{
    assert(context.is_section_pinned());
    auto data_pool = mempool(mem_data);
    if (data_pool.size)
    {
        try_set(data_, hrt::create(dt_uint8, { data_pool.size }, hrt::pool_shared));
    }

    rdata_ = context.section(".rdata");
    return ok();
}

result<uintptr_t> stackvm_runtime_module::reg(size_t id) const noexcept
{
    CHECK_WITH_ERR(id < regs_.size(), std::errc::result_out_of_range);
    return ok(regs_[id]);
}

result<void> stackvm_runtime_module::reg(size_t id, uintptr_t value) noexcept
{
    CHECK_WITH_ERR(id < regs_.size(), std::errc::result_out_of_range);
    regs_[id] = value;
    return ok();
}

result<runtime_shape_t> stackvm_runtime_module::shape_reg(size_t id) const noexcept
{
    CHECK_WITH_ERR(id < shape_regs_.size(), std::errc::result_out_of_range);
    return ok(shape_regs_[id]);
}

result<void> stackvm_runtime_module::shape_reg(size_t id, runtime_shape_t value) noexcept
{
    try
    {
        if (id >= shape_regs_.size())
            shape_regs_.resize(id + 1);
        shape_regs_[id] = std::move(value);
    }
    catch (...)
    {
        return err(std::errc::not_enough_memory);
    }

    return ok();
}

result<runtime_paddings_t> stackvm_runtime_module::paddings_reg(size_t id) const noexcept
{
    CHECK_WITH_ERR(id < paddings_regs_.size(), std::errc::result_out_of_range);
    return ok(paddings_regs_[id]);
}

result<void> stackvm_runtime_module::paddings_reg(size_t id, runtime_paddings_t value) noexcept
{
    try
    {
        if (id >= paddings_regs_.size())
            paddings_regs_.resize(id + 1);
        paddings_regs_[id] = std::move(value);
    }
    catch (...)
    {
        return err(std::errc::not_enough_memory);
    }

    return ok();
}

kernels::kernel_context &stackvm_runtime_module::kernel_context() noexcept
{
    return kernels::default_kernel_context();
}

result<std::unique_ptr<runtime_function>> stackvm_runtime_module::create_function() noexcept
{
    std::unique_ptr<runtime_function> mod(new (std::nothrow) stackvm_runtime_function(*this));
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

result<std::unique_ptr<runtime_module>> stackvm::create_stackvm_runtime_module()
{
    std::unique_ptr<runtime_module> mod(new (std::nothrow) stackvm_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}
