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

result<rt_module_activator_t>
find_runtime_activator(const module_kind_t &kind) {
    auto module_name = fmt::format("nncase.simulator.{}.dll", kind.data());
    auto mod = LoadLibraryExA(module_name.c_str(), nullptr,
                              LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!mod)
        mod = LoadLibraryA(module_name.c_str());
    TRY_WIN32_IF_NOT(mod);
    auto proc = GetProcAddress(mod, STR(RUNTIME_MODULE_ACTIVATOR_NAME));
    TRY_WIN32_IF_NOT(proc);
    return ok(reinterpret_cast<rt_module_activator_t>(proc));
}
#elif defined(__unix__) || defined(__APPLE__)
#ifdef __unix__
#define DYNLIB_EXT ".so"
#else
#define DYNLIB_EXT ".dylib"
#endif
#define TRY_POSIX_IF_NOT(x)                                                    \
    if (!(x)) {                                                                \
        return err(std::error_condition(1, std::system_category()));           \
    }

result<rt_module_activator_t>
find_runtime_activator(const module_kind_t &kind) {
    auto module_name =
        fmt::format("libnncase.simulator.{}" DYNLIB_EXT, kind.data());
    auto mod = dlopen(module_name.c_str(), RTLD_LAZY);
    TRY_POSIX_IF_NOT(mod);
    auto proc = dlsym(mod, STR(RUNTIME_MODULE_ACTIVATOR_NAME));
    TRY_POSIX_IF_NOT(proc);
    return ok(reinterpret_cast<rt_module_activator_t>(proc));
}
#else
#define NNCASE_NO_LOADABLE_RUNTIME
#endif
} // namespace
#else
#define NNCASE_NO_LOADABLE_RUNTIME

result<rt_module_activator_t>
find_runtime_activator(const module_kind_t &kind) {
    auto builtin_runtime = builtin_runtimes;
    while (builtin_runtime->activator) {
        if (!strcmp(kind.data(), builtin_runtime->id.data())) {
            return ok(builtin_runtime->activator);
        }
    }

    return err(nncase_errc::runtime_not_found);
}
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

#ifdef NNCASE_DEFAULT_BUILTIN_RUNTIMES
runtime_registration nncase::runtime::builtin_runtimes[] = {{}};
#endif
