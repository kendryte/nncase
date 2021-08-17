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
#include "model.h"
#include "result.h"
#include "runtime_function.h"
#include "runtime_tensor.h"

BEGIN_NS_NNCASE_RUNTIME

class interpreter;

struct NNCASE_API runtime_module_init_context
{
    virtual bool is_section_pinned() const noexcept = 0;
    virtual interpreter &interp() noexcept = 0;
    virtual const module_header &header() noexcept = 0;
    virtual gsl::span<const gsl::byte> section(const char *name) noexcept = 0;
};

class NNCASE_API runtime_module
{
public:
    static result<std::unique_ptr<runtime_module>> create(const module_type_t &type);

    runtime_module() = default;
    runtime_module(const runtime_module &) = delete;
    virtual ~runtime_module() = default;
    runtime_module &operator=(const runtime_module &) = delete;

    result<void> initialize(gsl::span<const gsl::byte> payload, interpreter &interp) noexcept;
    const module_type_t &type() const noexcept;

    interpreter &interp() const noexcept { return *interp_; }

    uint32_t mempools_size() const noexcept;
    const mempool_desc &mempool(size_t index) const noexcept;
    mempool_desc mempool(memory_location_t location) const noexcept;

    result<runtime_function *> find_function_by_id(size_t index) noexcept;

protected:
    virtual result<void> initialize_before_functions(runtime_module_init_context &context) noexcept;
    virtual result<void> initialize_after_functions(runtime_module_init_context &context) noexcept;
    virtual result<std::unique_ptr<runtime_function>> create_function() noexcept = 0;

    gsl::span<std::unique_ptr<runtime_function>> functions() noexcept { return functions_; }

private:
    module_header header_;
    std::vector<mempool_desc> mempools_;
    std::vector<mempool_desc> shared_mempools_;
    std::vector<std::unique_ptr<runtime_function>> functions_;
    interpreter *interp_ = nullptr;
};

END_NS_NNCASE_RUNTIME
