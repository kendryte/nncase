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
#include "runtime_section_context.h"
#include <nncase/runtime/stream_reader.h>
#include <nncase/type.h>
#include <nncase/value.h>

BEGIN_NS_NNCASE_RUNTIME

class interpreter;
class runtime_module;
struct runtime_module_init_context;

struct NNCASE_API runtime_function_init_context
    : public runtime_section_context {
    virtual runtime_module_init_context &module_init_context() noexcept = 0;
    virtual const function_header &header() noexcept = 0;
};

class NNCASE_API runtime_function {
  public:
    runtime_function(runtime_module &rt_module);
    runtime_function(const runtime_function &) = delete;
    virtual ~runtime_function() = default;
    runtime_function &operator=(const runtime_function &) = delete;

    result<void>
    initialize(gsl::span<const gsl::byte> payload,
               runtime_module_init_context &module_init_context) noexcept;
    result<void>
    initialize(stream_reader &reader,
               runtime_module_init_context &module_init_context) noexcept;

    runtime_module &module() const noexcept;

    uint32_t parameters_size() const noexcept;
    result<type> parameter_type(size_t index) const noexcept;
    const type &return_type() const noexcept;

    result<value_t> invoke(gsl::span<value_t> parameters,
                           value_t return_value = nullptr) noexcept;

  protected:
    virtual result<void>
    initialize_core(runtime_function_init_context &context) noexcept = 0;

    virtual result<value_t> invoke_core(gsl::span<value_t> parameters,
                                        value_t return_value) noexcept = 0;

  private:
    function_header header_;
    runtime_module &rt_module_;
    std::vector<type> parameter_types_;
    type return_type_;
};

END_NS_NNCASE_RUNTIME
