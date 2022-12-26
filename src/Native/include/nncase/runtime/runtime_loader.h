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
#include <memory>
#include <nncase/runtime/error.h>
#include <nncase/runtime/model.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/runtime_module.h>

BEGIN_NS_NNCASE_RUNTIME

typedef void (*rt_module_activator_t)(
    result<std::unique_ptr<runtime_module>> &result);
typedef void (*rt_module_collector_t)(
    result<
        std::vector<std::pair<std::string, runtime_module::custom_call_type>>>
        &result);

#define RUNTIME_MODULE_ACTIVATOR_NAME create_runtime_module
#define RUNTIME_MODULE_COLLECTOR_NAME collect_custom_call

struct runtime_registration {
    module_kind_t id;
    rt_module_activator_t activator;
    rt_module_collector_t collector;
};

extern runtime_registration builtin_runtimes[];

END_NS_NNCASE_RUNTIME
