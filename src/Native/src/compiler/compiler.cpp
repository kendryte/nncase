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
#include <absl/debugging/failure_signal_handler.h>
#include <filesystem>
#include <fstream>
#include <hostfxr.h>
#include <nethost.h>
#include <nncase/compiler.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>

#ifdef WIN32
#include <Windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#define THROW_IF_NOT(x, err)                                                   \
    if (!(x)) {                                                                \
        throw std::system_error(make_error_code(err));                         \
    }

#define UNMANAGEDCALLERSONLY_METHOD ((const char_t *)-1)

namespace {
struct c_api_mt {
    clr_object_handle_t (*array_create)(nncase_array_element_kind_t kind,
                                        const clr_object_handle_t *elements,
                                        size_t count);
    clr_object_handle_t (*array_get_item)(clr_object_handle_t array,
                                          size_t index);
    size_t (*array_get_length)(clr_object_handle_t array);
    clr_object_handle_t (*calibration_dataset_provider_create)(
        clr_object_handle_t dataset, size_t samplesCount,
        clr_object_handle_t fn_params);
    void (*handle_free)(clr_object_handle_t handle);
    clr_object_handle_t (*compile_options_create)();
    void (*compile_options_set_input_file)(clr_object_handle_t compile_options,
                                           const char *input_file,
                                           size_t input_file_length);
    void (*compile_options_set_input_format)(
        clr_object_handle_t compile_options, const char *input_format,
        size_t input_format_length);
    void (*compile_options_set_target)(clr_object_handle_t compile_options,
                                       const char *target,
                                       size_t target_length);
    void (*compile_options_set_dump_level)(clr_object_handle_t compile_options,
                                           int32_t dump_level);
    void (*compile_options_set_dump_dir)(clr_object_handle_t compile_options,
                                         const char *dump_dir,
                                         size_t dump_dir_length);
    void (*compile_options_set_quantize_options)(
        clr_object_handle_t compile_options,
        clr_object_handle_t quantize_options);
    void (*compile_options_set_quant_type)(clr_object_handle_t compile_options,
                                           clr_object_handle_t quant_type);
    void (*compile_options_set_quant_mode)(clr_object_handle_t compile_options,
                                           nncase_quant_mode_t quant_mode);
    void (*compiler_initialize)();
    clr_object_handle_t (*compiler_create)(clr_object_handle_t compile_options);
    clr_object_handle_t (*compiler_import_module)(clr_object_handle_t compiler,
                                                  clr_object_handle_t stream);
    void (*compiler_compile)(clr_object_handle_t compiler);
    void (*compiler_gencode)(clr_object_handle_t compiler,
                             clr_object_handle_t stream);
    clr_object_handle_t (*datatype_from_typecode)(nncase::typecode_t typecode);
    clr_object_handle_t (*expr_evaluate)(clr_object_handle_t expr,
                                         clr_object_handle_t parameters,
                                         clr_object_handle_t inputs);
    clr_object_handle_t (*function_get_body)(clr_object_handle_t function);
    clr_object_handle_t (*function_get_parameters)(
        clr_object_handle_t function);
    clr_object_handle_t (*ir_module_get_entry)(clr_object_handle_t module);
    void (*luanch_debugger)();
    clr_object_handle_t (*quantize_options_create)();
    void (*quantize_options_set_calibration_dataset)(
        clr_object_handle_t quantize_options, clr_object_handle_t dataset);
    void (*quantize_options_set_calibration_method)(
        clr_object_handle_t quantize_options, nncase_calib_method_t method);
    clr_object_handle_t (*rtvalue_from_handle)(nncase::value_node *value);
    nncase::value_node *(*rtvalue_get_handle)(clr_object_handle_t rtvalue);
    clr_object_handle_t (*stream_create)(const nncase_stream_mt_t *mt,
                                         void *handle);
    bool (*target_exists)(const char *target_name, size_t target_name_length);
};

typedef int (*get_function_pointer_fn)(const char_t *type_name,
                                       const char_t *method_name,
                                       const char_t *delegate_type_name,
                                       void *load_context, void *reserved,
                                       /*out*/ void **delegate);

typedef void (*c_api_initialize_fn)(c_api_mt *mt);

#ifdef WIN32
#define THROW_WIN32_IF_NOT(x)                                                  \
    if (!(x)) {                                                                \
        throw std::system_error(GetLastError(), std::system_category());       \
    }

HMODULE load_library(const char_t *name) {
    auto mod = LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!mod)
        mod = LoadLibraryW(name);
    THROW_WIN32_IF_NOT(mod);
    return mod;
}

FARPROC load_symbol(HMODULE module, const char *name) {
    auto symbol = GetProcAddress(module, name);
    THROW_WIN32_IF_NOT(symbol);
    return symbol;
}

#define _T(x) L##x
#elif defined(__unix__) || defined(__APPLE__)
#define THROW_DL_IF_NOT(x)                                                     \
    if (!(x)) {                                                                \
        throw std::system_error(errno, std::system_category(), dlerror());     \
    }

void *load_library(const char_t *name) {
    auto mod = dlopen(name, RTLD_LAZY);
    THROW_DL_IF_NOT(mod);
    return mod;
}

void *load_symbol(void *module, const char *name) {
    auto symbol = dlsym(module, name);
    THROW_DL_IF_NOT(symbol);
    return symbol;
}

#define _T(x) x
#endif

struct premain {
    premain() {
        absl::FailureSignalHandlerOptions failure_signal_handler_options;
        failure_signal_handler_options.symbolize_stacktrace = true;
        failure_signal_handler_options.use_alternate_stack = true;
        failure_signal_handler_options.alarm_on_failure_secs = 5;
        failure_signal_handler_options.call_previous_handler = true;
        absl::InstallFailureSignalHandler(failure_signal_handler_options);
    }
} premain_v;

c_api_initialize_fn
load_compiler_c_api_initializer(const char *root_assembly_path) {
    size_t path_length;
    if (get_hostfxr_path(nullptr, &path_length, nullptr) != 0x80008098)
        throw std::runtime_error("Failed to get hostfxr path.");

    std::basic_string<char_t> path(path_length, '\0');
    if (get_hostfxr_path(path.data(), &path_length, nullptr))
        throw std::runtime_error("Failed to get hostfxr path.");

    auto hostfxr_mod = load_library(path.c_str());
    auto hostfxr_initialize =
        (hostfxr_initialize_for_dotnet_command_line_fn)load_symbol(
            hostfxr_mod, "hostfxr_initialize_for_dotnet_command_line");

    hostfxr_handle handle;
    std::filesystem::path compiler_path(root_assembly_path);
    const char_t *args[] = {compiler_path.c_str()};
    hostfxr_initialize(1, args, nullptr, &handle);
    THROW_IF_NOT(handle, nncase::runtime::nncase_errc::runtime_not_found);

    auto hostfxr_get_delegate = (hostfxr_get_runtime_delegate_fn)load_symbol(
        hostfxr_mod, "hostfxr_get_runtime_delegate");

    get_function_pointer_fn hostfxr_get_fn_ptr;
    hostfxr_get_delegate(handle, hdt_get_function_pointer,
                         (void **)&hostfxr_get_fn_ptr);
    THROW_IF_NOT(hostfxr_get_fn_ptr,
                 nncase::runtime::nncase_errc::runtime_not_found);

    c_api_initialize_fn c_api_initialize;
    hostfxr_get_fn_ptr(_T("Nncase.Compiler.Interop.CApi, Nncase.Compiler"),
                       _T("Initialize"), UNMANAGEDCALLERSONLY_METHOD, nullptr,
                       nullptr, (void **)&c_api_initialize);
    THROW_IF_NOT(c_api_initialize,
                 nncase::runtime::nncase_errc::runtime_not_found);
    return c_api_initialize;
}

c_api_mt g_c_api_mt;
} // namespace

int nncase_clr_initialize(const char *root_assembly_path) {
    if (!g_c_api_mt.handle_free) {
        auto init = load_compiler_c_api_initializer(root_assembly_path);
        init(&g_c_api_mt);
        g_c_api_mt.compiler_initialize();
    }

    return 0;
}

int nncase_clr_array_create(nncase_array_element_kind_t kind,
                            const clr_object_handle_t *elements, size_t count,
                            clr_object_handle_t *array) {
    *array = g_c_api_mt.array_create(kind, elements, count);
    return 0;
}

int nncase_clr_array_get_item(clr_object_handle_t array, size_t index,
                              clr_object_handle_t *item) {
    *item = g_c_api_mt.array_get_item(array, index);
    return 0;
}

int nncase_clr_array_get_length(clr_object_handle_t array, size_t *length) {
    *length = g_c_api_mt.array_get_length(array);
    return 0;
}

int nncase_clr_calibration_dataset_provider_create(
    clr_object_handle_t dataset, size_t samplesCount,
    clr_object_handle_t fn_params, clr_object_handle_t *provider) {
    *provider = g_c_api_mt.calibration_dataset_provider_create(
        dataset, samplesCount, fn_params);
    return 0;
}

int nncase_clr_handle_free(clr_object_handle_t handle) {
    g_c_api_mt.handle_free(handle);
    return 0;
}

int nncase_clr_compile_options_create(clr_object_handle_t *compile_options) {
    *compile_options = g_c_api_mt.compile_options_create();
    return 0;
}

int nncase_clr_compile_options_set_inputfile(
    clr_object_handle_t compile_options, const char *input_file,
    size_t input_file_length) {
    g_c_api_mt.compile_options_set_input_file(compile_options, input_file,
                                              input_file_length);
    return 0;
}

int nncase_clr_compile_options_set_input_format(
    clr_object_handle_t compile_options, const char *input_format,
    size_t input_format_length) {
    g_c_api_mt.compile_options_set_input_format(compile_options, input_format,
                                                input_format_length);
    return 0;
}

int nncase_clr_compile_options_set_target(clr_object_handle_t compile_options,
                                          const char *target,
                                          size_t target_length) {
    g_c_api_mt.compile_options_set_target(compile_options, target,
                                          target_length);
    return 0;
}

int nncase_clr_compile_options_set_dump_level(
    clr_object_handle_t compile_options, int32_t dump_level) {
    g_c_api_mt.compile_options_set_dump_level(compile_options, dump_level);
    return 0;
}

int nncase_clr_compile_options_set_dump_dir(clr_object_handle_t compile_options,
                                            const char *dump_dir,
                                            size_t dump_dir_length) {
    g_c_api_mt.compile_options_set_dump_dir(compile_options, dump_dir,
                                            dump_dir_length);
    return 0;
}

int nncase_clr_compile_options_set_quantize_options(
    clr_object_handle_t compile_options, clr_object_handle_t quantize_options) {
    g_c_api_mt.compile_options_set_quantize_options(compile_options,
                                                    quantize_options);
    return 0;
}

int nncase_clr_compile_options_set_quant_mode(
    clr_object_handle_t compile_options, nncase_quant_mode_t quant_mode) {
    g_c_api_mt.compile_options_set_quant_mode(compile_options, quant_mode);
    return 0;
}

int nncase_clr_compiler_create(clr_object_handle_t compile_options,
                               clr_object_handle_t *compiler) {
    *compiler = g_c_api_mt.compiler_create(compile_options);
    return 0;
}

int nncase_clr_compiler_import_module(clr_object_handle_t compiler,
                                      clr_object_handle_t stream,
                                      clr_object_handle_t *module) {
    *module = g_c_api_mt.compiler_import_module(compiler, stream);
    return 0;
}

int nncase_clr_compiler_compile(clr_object_handle_t compiler) {
    g_c_api_mt.compiler_compile(compiler);
    return 0;
}

int nncase_clr_compiler_gencode(clr_object_handle_t compiler,
                                clr_object_handle_t stream) {
    g_c_api_mt.compiler_gencode(compiler, stream);
    return 0;
}

int nncase_clr_datatype_from_typecode(nncase::typecode_t typecode,
                                      clr_object_handle_t *datatype) {
    *datatype = g_c_api_mt.datatype_from_typecode(typecode);
    return 0;
}

int nncase_clr_expr_evaluate(clr_object_handle_t expr,
                             clr_object_handle_t inputs,
                             clr_object_handle_t parameters,
                             clr_object_handle_t *result) {
    *result = g_c_api_mt.expr_evaluate(expr, parameters, inputs);
    return 0;
}

int nncase_clr_function_get_body(clr_object_handle_t function,
                                 clr_object_handle_t *body) {
    *body = g_c_api_mt.function_get_body(function);
    return 0;
}

int nncase_clr_function_get_parameters(clr_object_handle_t function,
                                       clr_object_handle_t *parameters) {
    *parameters = g_c_api_mt.function_get_parameters(function);
    return 0;
}

int nncase_clr_ir_module_get_entry(clr_object_handle_t module,
                                   clr_object_handle_t *entry) {
    *entry = g_c_api_mt.ir_module_get_entry(module);
    return 0;
}

int nncase_clr_launch_debugger() {
    g_c_api_mt.luanch_debugger();
    return 0;
}

int nncase_clr_quantize_options_create(clr_object_handle_t *quantize_options) {
    *quantize_options = g_c_api_mt.quantize_options_create();
    return 0;
}

int nncase_clr_quantize_options_set_calibration_dataset(
    clr_object_handle_t quantize_options, clr_object_handle_t dataset) {
    g_c_api_mt.quantize_options_set_calibration_dataset(quantize_options,
                                                        dataset);
    return 0;
}

int nncase_clr_quantize_options_set_calibration_method(
    clr_object_handle_t quantize_options, nncase_calib_method_t method) {
    g_c_api_mt.quantize_options_set_calibration_method(quantize_options,
                                                       method);
    return 0;
}

int nncase_clr_rtvalue_from_handle(nncase::value_node *value,
                                   clr_object_handle_t *rtvalue) {
    *rtvalue = g_c_api_mt.rtvalue_from_handle(nncase::value_t(value).detach());
    return 0;
}

int nncase_clr_rtvalue_get_handle(clr_object_handle_t rtvalue,
                                  nncase::value_node **value) {
    nncase::value_t v(g_c_api_mt.rtvalue_get_handle(rtvalue));
    *value = v.detach();
    return 0;
}

int nncase_clr_stream_create(const nncase_stream_mt_t *mt, void *handle,
                             clr_object_handle_t *stream) {
    *stream = g_c_api_mt.stream_create(mt, handle);
    return 0;
}

bool nncase_clr_target_exists(const char *target_name,
                              size_t target_name_length) {
    return g_c_api_mt.target_exists(target_name, target_name_length);
}
