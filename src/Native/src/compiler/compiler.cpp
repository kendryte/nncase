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
#include <Windows.h>
#include <filesystem>
#include <hostfxr.h>
#include <nethost.h>
#include <nncase/compiler.h>

#define THROW_WIN32_IF_NOT(x)                                                  \
    if (!(x)) {                                                                \
        throw std::system_error(GetLastError(), std::system_category());       \
    }

#define THROW_IF_NOT(x)                                                        \
    if (!(x)) {                                                                \
        throw std::system_error(GetLastError(), std::system_category());       \
    }

#define UNMANAGEDCALLERSONLY_METHOD ((const char_t *)-1)

namespace {
struct c_api_mt {
    void (*clr_handle_free)(clr_object_handle handle);
    clr_object_handle (*compile_options_create)();
    void (*compile_options_set_input_file)(clr_object_handle compile_options,
                                           const char *input_file,
                                           size_t input_file_length);
    void (*compile_options_set_input_format)(clr_object_handle compile_options,
                                             const char *input_format,
                                             size_t input_format_length);
    void (*compile_options_set_target)(clr_object_handle compile_options,
                                       const char *target,
                                       size_t target_length);
    void (*compile_options_set_dump_level)(clr_object_handle compile_options,
                                           int32_t dump_level);
    intptr_t reserved[3];
    void (*compiler_initialize)();
};

typedef int (*get_function_pointer_fn)(const char_t *type_name,
                                       const char_t *method_name,
                                       const char_t *delegate_type_name,
                                       void *load_context, void *reserved,
                                       /*out*/ void **delegate);

typedef void (*c_api_initialize_fn)(c_api_mt *mt);

c_api_initialize_fn
load_compiler_c_api_initializer(const char *root_assembly_path) {
    size_t path_length;
    if (get_hostfxr_path(nullptr, &path_length, nullptr) != 0x80008098)
        throw std::runtime_error("Failed to get hostfxr path.");

    std::basic_string<char_t> path(path_length, '\0');
    if (get_hostfxr_path(path.data(), &path_length, nullptr))
        throw std::runtime_error("Failed to get hostfxr path.");

    auto hostfxr_mod = LoadLibraryW(path.c_str());
    THROW_WIN32_IF_NOT(hostfxr_mod);

    auto hostfxr_initialize =
        (hostfxr_initialize_for_dotnet_command_line_fn)GetProcAddress(
            hostfxr_mod, "hostfxr_initialize_for_dotnet_command_line");
    THROW_WIN32_IF_NOT(hostfxr_initialize);

    hostfxr_handle handle;
    std::filesystem::path compiler_path(root_assembly_path);
    const char_t *args[] = {compiler_path.c_str()};
    hostfxr_initialize(1, args, nullptr, &handle);
    THROW_IF_NOT(handle);

    auto hostfxr_get_delegate = (hostfxr_get_runtime_delegate_fn)GetProcAddress(
        hostfxr_mod, "hostfxr_get_runtime_delegate");
    THROW_WIN32_IF_NOT(hostfxr_get_delegate);

    get_function_pointer_fn hostfxr_get_fn_ptr;
    hostfxr_get_delegate(handle, hdt_get_function_pointer,
                         (void **)&hostfxr_get_fn_ptr);
    THROW_WIN32_IF_NOT(hostfxr_get_fn_ptr);

    c_api_initialize_fn c_api_initialize;
    hostfxr_get_fn_ptr(L"Nncase.Compiler.Interop.CApi, Nncase.Compiler",
                       L"Initialize", UNMANAGEDCALLERSONLY_METHOD, nullptr,
                       nullptr, (void **)&c_api_initialize);
    THROW_WIN32_IF_NOT(c_api_initialize);
    return c_api_initialize;
}

c_api_mt g_c_api_mt;
} // namespace

int nncase_compiler_initialize(const char *root_assembly_path) {
    if (!g_c_api_mt.clr_handle_free) {
        auto init = load_compiler_c_api_initializer(root_assembly_path);
        init(&g_c_api_mt);
        g_c_api_mt.compiler_initialize();
    }

    return 0;
}

int nncase_compiler_compile_options_create(clr_object_handle *handle) {
    *handle = g_c_api_mt.compile_options_create();
    return 0;
}

int nncase_compiler_compile_options_set_inputfile(
    clr_object_handle compile_options, const char *input_file,
    size_t input_file_length) {
    g_c_api_mt.compile_options_set_input_file(compile_options, input_file,
                                              input_file_length);
    return 0;
}

int nncase_compiler_compile_options_set_input_format(
    clr_object_handle compile_options, const char *input_format,
    size_t input_format_length) {
    g_c_api_mt.compile_options_set_input_format(compile_options, input_format,
                                                input_format_length);
    return 0;
}

int nncase_compiler_compile_options_set_target(
    clr_object_handle compile_options, const char *target,
    size_t target_length) {
    g_c_api_mt.compile_options_set_target(compile_options, target,
                                          target_length);
    return 0;
}

int nncase_compiler_compile_options_set_dump_level(
    clr_object_handle compile_options, int32_t dump_level) {
    g_c_api_mt.compile_options_set_dump_level(compile_options, dump_level);
    return 0;
}
