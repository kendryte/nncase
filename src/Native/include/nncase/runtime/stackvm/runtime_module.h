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
#include "../runtime_module.h"

BEGIN_NS_NNCASE_RT_MODULE(stackvm)

NNCASE_INLINE_VAR constexpr module_kind_t stackvm_module_kind =
    to_module_kind("stackvm");
NNCASE_INLINE_VAR constexpr uint32_t stackvm_module_version = 1;

NNCASE_API result<std::unique_ptr<runtime_module>>
create_stackvm_runtime_module();

NNCASE_API result<
    std::vector<std::pair<std::string, runtime_module::custom_call_type>>>
create_stackvm_custom_calls();

END_NS_NNCASE_RT_MODULE
