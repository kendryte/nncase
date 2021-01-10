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
#ifdef WIN32
#include <Windows.h>
#endif

#include "runtime_loader.h"
#include <fmt/format.h>
#include <nncase/runtime/runtime.h>

using namespace nncase;
using namespace nncase::runtime;

#define STR_(x) #x
#define STR(x) STR_(x)

namespace
{
template <std::size_t N, std::size_t... Is>
constexpr model_target_t
to_target_id(const char (&a)[N], std::index_sequence<Is...>)
{
    return { { a[Is]... } };
}

template <std::size_t N>
constexpr model_target_t to_target_id(const char (&a)[N])
{
    return to_target_id(a, std::make_index_sequence<N>());
}
}

#ifndef NNCASE_SIMULATOR
// builtin runtime
#include <builtin_runtimes.inl>
#endif

namespace
{
#ifdef WIN32
#define TRY_WIN32_IF_NOT(x)                                                       \
    if (!(x))                                                                     \
    {                                                                             \
        return err(std::error_condition(GetLastError(), std::system_category())); \
    }

result<runtime_activator_t> find_runtime_activator(const model_target_t &target_id)
{
#ifdef NNCASE_SIMULATOR
    auto module_name = fmt::format("nncase.targets.{}.dll", target_id.data());
#else
    auto module_name = fmt::format("nncase.runtime.{}.dll", target_id.data());
#endif
    auto mod = LoadLibraryA(module_name.c_str());
    TRY_WIN32_IF_NOT(mod);
#ifdef NNCASE_SIMULATOR
    auto proc = GetProcAddress(mod, STR(SIMULATOR_ACTIVATOR_NAME));
#else
    auto proc = GetProcAddress(mod, STR(RUNTIME_ACTIVATOR_NAME));
#endif
    TRY_WIN32_IF_NOT(proc);
    return reinterpret_cast<runtime_activator_t>(proc);
}
#else
#define NNCASE_NO_LOADABLE_RUNTIME
#endif
}

runtime_base::~runtime_base()
{
}

void runtime::create_runtime(const model_target_t &target_id, result<std::unique_ptr<runtime_base>> &result)
{
#ifndef NNCASE_SIMULATOR
    for (auto &reg : builtin_runtimes)
    {
        if (!strcmp(target_id.data(), reg.id.data()))
        {
            return reg.activator();
        }
    }
#endif

#ifndef NNCASE_NO_LOADABLE_RUNTIME
    try_var_err(activator, find_runtime_activator(target_id), result);
    activator(result);
#else
    result = err(nncase_errc::runtime_not_found);
#endif
}
