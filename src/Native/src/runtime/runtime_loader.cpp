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
#ifdef WIN32
#include <Windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <cstring>
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_module.h>
#include <nncase/runtime/stackvm/runtime_module.h>

using namespace nncase;
using namespace nncase::runtime;

#define STR_(x) #x
#define STR(x) STR_(x)

#ifdef NNCASE_SIMULATOR
#include <fmt/format.h>

namespace {
#ifdef WIN32
#define TRY_WIN32_IF_NOT(x)                                                    \
    if (!(x)) {                                                                \
        return err(                                                            \
            std::error_condition(GetLastError(), std::system_category()));     \
    }
#define FindRuntimeMethod(snake_name, upper_name)                              \
    result<rt_module_##snake_name##_t> find_runtime_##snake_name(              \
        const module_kind_t &kind) {                                           \
        auto module_name =                                                     \
            fmt::format("nncase.simulator.{}.dll", kind.data());               \
        auto mod = LoadLibraryExA(module_name.c_str(), nullptr,                \
                                  LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);           \
        if (!mod)                                                              \
            mod = LoadLibraryA(module_name.c_str());                           \
        TRY_WIN32_IF_NOT(mod);                                                 \
        auto proc =                                                            \
            GetProcAddress(mod, STR(RUNTIME_MODULE_##upper_name##_NAME));      \
        TRY_WIN32_IF_NOT(proc);                                                \
        return ok(reinterpret_cast<rt_module_##snake_name##_t>(proc));         \
    }

// clang-format off
FindRuntimeMethod(activator, ACTIVATOR)
FindRuntimeMethod(collector, COLLECTOR)
// clang-format on

#undef FindRuntimeMethod

#elif defined(__unix__) || defined(__APPLE__)
#ifdef __unix__
#define DYNLIB_EXT ".so"
#else
#define DYNLIB_EXT ".dylib"
#endif

#define FindRuntimeMethod(snake_name, upper_name)                              \
    result<rt_module_##snake_name##_t> find_runtime_##snake_name(              \
        const module_kind_t &kind) {                                           \
        auto module_name =                                                     \
            fmt::format("libnncase.simulator.{}" DYNLIB_EXT, kind.data());     \
        auto mod = dlopen(module_name.c_str(), RTLD_LAZY);                     \
        if (!(mod))                                                            \
            return err(nncase_errc::runtime_not_found);                        \
        auto proc = dlsym(mod, STR(RUNTIME_MODULE_##upper_name##_NAME));       \
        if (!(proc))                                                           \
            return err(nncase_errc::runtime_register_not_found);               \
        return ok(reinterpret_cast<rt_module_##snake_name##_t>(proc));         \
    }
// clang-format off
FindRuntimeMethod(activator, ACTIVATOR) 
FindRuntimeMethod(collector, COLLECTOR)
// clang-format on

#undef FindRuntimeMethod

#else
#define NNCASE_NO_LOADABLE_RUNTIME
#endif
} // namespace
#else
#define NNCASE_NO_LOADABLE_RUNTIME

#define FindRuntimeMethod(snake_name)                                          \
    result<rt_module_##snake_name##_t> find_runtime_##snake_name(              \
        const module_kind_t &kind) {                                           \
        auto builtin_runtime = builtin_runtimes;                               \
        while (builtin_runtime->snake_name) {                                  \
            if (!strcmp(kind.data(), builtin_runtime->id.data())) {            \
                return ok(builtin_runtime->snake_name);                        \
            }                                                                  \
        }                                                                      \
        return err(nncase_errc::runtime_not_found);                            \
    }
// clang-format off
FindRuntimeMethod(activator)
FindRuntimeMethod(collector)
// clang-format on
#undef FindRuntimeMethod
#endif

result<std::unique_ptr<runtime_module>>
runtime_module::create(const module_kind_t &kind) {
    if (!strncmp(kind.data(), stackvm::stackvm_module_kind.data(),
                 MAX_MODULE_KIND_LENGTH))
        return stackvm::create_stackvm_runtime_module();

    result<std::unique_ptr<runtime_module>> rt_module(
        nncase_errc::runtime_not_found);
    try_var(activator, find_runtime_activator(kind));
    activator(rt_module);
    return rt_module;
}

result<std::vector<std::pair<std::string, runtime_module::custom_call_type>>>
runtime_module::collect(const module_kind_t &kind) {
    if (!strncmp(kind.data(), stackvm::stackvm_module_kind.data(),
                 MAX_MODULE_KIND_LENGTH))
        return stackvm::create_stackvm_custom_calls();

    result<
        std::vector<std::pair<std::string, runtime_module::custom_call_type>>>
        table(nncase_errc::runtime_not_found);
    try_var(collector, find_runtime_collector(kind));
    collector(table);
    return table;
}

#ifdef NNCASE_DEFAULT_BUILTIN_RUNTIMES
runtime_registration nncase::runtime::builtin_runtimes[] = {{}};
#endif
