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
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#else
#error "Unsupported platform"
#endif

#include <fmt/format.h>
#include <nncase/plugin_loader.h>
#include <nncase/targets/neutral_target.h>

using namespace nncase;
using namespace nncase::plugin_loader;

#define STR_(x) #x
#define STR(x) STR_(x)

namespace
{
#ifdef WIN32
#define THROW_WIN32_IF_NOT(x, fmt_str, ...)                                                                    \
    if (!(x))                                                                                                  \
    {                                                                                                          \
        auto err_code = GetLastError();                                                                        \
        auto err_msg = std::system_category().message(err_code);                                               \
        throw std::system_error(err_code, std::system_category(), fmt::format(fmt_str, err_msg, __VA_ARGS__)); \
    }

target_activator_t find_target_activator(std::string_view name)
{
    auto module_name = fmt::format("nncase.targets.{}.dll", name);
    auto mod = LoadLibraryExA(module_name.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    THROW_WIN32_IF_NOT(mod, "Cannot load module: {1}, {0}", module_name);
    auto proc = GetProcAddress(mod, STR(TARGET_ACTIVATOR_NAME));
    THROW_WIN32_IF_NOT(proc, "Cannot load proc \"{1}\" in module: {2}, {0}", STR(TARGET_ACTIVATOR_NAME), module_name);
    return reinterpret_cast<target_activator_t>(proc);
}
#elif defined(__unix__) || defined(__APPLE__)
#ifdef __unix__
#define DYNLIB_EXT ".so"
#else
#define DYNLIB_EXT ".dylib"
#endif
#define THROW_DLERR_IF_NOT(x, fmt_str, ...)                                   \
    if (!(x))                                                                 \
    {                                                                         \
        auto err_msg = dlerror();                                             \
        throw std::runtime_error(fmt::format(fmt_str, err_msg, __VA_ARGS__)); \
    }

target_activator_t find_target_activator(std::string_view name)
{
    auto module_name = fmt::format("libnncase.targets.{}" DYNLIB_EXT, name);
    auto mod = dlopen(module_name.c_str(), RTLD_LAZY);
    THROW_DLERR_IF_NOT(mod, "Cannot load module: {1}, {0}", module_name);
    auto proc = dlsym(mod, STR(TARGET_ACTIVATOR_NAME));
    THROW_DLERR_IF_NOT(proc, "Cannot load proc \"{1}\" in module: {2}, {0}", STR(TARGET_ACTIVATOR_NAME), module_name);
    return reinterpret_cast<target_activator_t>(proc);
}
#endif
}

std::unique_ptr<target> plugin_loader::create_target(std::string_view name)
{
    auto activator = find_target_activator(name);
    return std::unique_ptr<target>(activator());
}
