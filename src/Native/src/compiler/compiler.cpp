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
// clang-format off
#include <Windows.h>
#include <DbgHelp.h>
// clang-format on
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#define THROW_IF_NOT(x, format, ...)                                           \
    if (!(x)) {                                                                \
        char msg[256];                                                         \
        std::snprintf(msg, std::size(msg), format, __VA_ARGS__);               \
        throw std::runtime_error(msg);                                         \
    }

#define UNMANAGEDCALLERSONLY_METHOD ((const char_t *)-1)

namespace {

typedef int (*get_function_pointer_fn)(const char_t *type_name,
                                       const char_t *method_name,
                                       const char_t *delegate_type_name,
                                       void *load_context, void *reserved,
                                       /*out*/ void **delegate);

typedef int (*load_assembly_and_get_function_pointer_fn)(
    const char_t *assembly_path, const char_t *type_name,
    const char_t *method_name, const char_t *delegate_type_name, void *reserved,
    /*out*/ void **delegate);

typedef void (*c_api_initialize_fn)(nncase_api_mt_t *mt);

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

#if 0

inline BOOL IsDataSectionNeeded(const WCHAR *pModuleName) {
    if (pModuleName == 0) {
        return FALSE;
    }

    WCHAR szFileName[_MAX_FNAME] = L"";
    _wsplitpath(pModuleName, NULL, NULL, szFileName, NULL);

    if (_wcsicmp(szFileName, L"ntdll") == 0)
        return TRUE;

    return FALSE;
}

inline BOOL CALLBACK MiniDumpCallback(PVOID pParam,
                                      const PMINIDUMP_CALLBACK_INPUT pInput,
                                      PMINIDUMP_CALLBACK_OUTPUT pOutput) {
    if (pInput == 0 || pOutput == 0)
        return FALSE;

    switch (pInput->CallbackType) {
    case ModuleCallback:
        //if (pOutput->ModuleWriteFlags & ModuleWriteDataSeg)
        //    if (!IsDataSectionNeeded(pInput->Module.FullPath))
        //        pOutput->ModuleWriteFlags &= (~ModuleWriteDataSeg);
    case IncludeModuleCallback:
    case IncludeThreadCallback:
    case ThreadCallback:
    case ThreadExCallback:
        return TRUE;
    default:;
    }

    return FALSE;
}

inline void CreateMiniDump(PEXCEPTION_POINTERS pep, LPCTSTR strFileName) {
    HANDLE hFile =
        CreateFile(strFileName, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_WRITE,
                   NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if ((hFile != NULL) && (hFile != INVALID_HANDLE_VALUE)) {
        MINIDUMP_EXCEPTION_INFORMATION mdei;
        mdei.ThreadId = GetCurrentThreadId();
        mdei.ExceptionPointers = pep;
        mdei.ClientPointers = NULL;

        MINIDUMP_CALLBACK_INFORMATION mci;
        mci.CallbackRoutine = (MINIDUMP_CALLBACK_ROUTINE)MiniDumpCallback;
        mci.CallbackParam = 0;

        ::MiniDumpWriteDump(::GetCurrentProcess(), ::GetCurrentProcessId(),
                            hFile, MiniDumpNormal, (pep != 0) ? &mdei : 0, NULL,
                            &mci);

        CloseHandle(hFile);
    }
}

LONG __stdcall MyUnhandledExceptionFilter(PEXCEPTION_POINTERS pExceptionInfo) {
    CreateMiniDump(pExceptionInfo, "core.dmp");
    return EXCEPTION_EXECUTE_HANDLER;
}

void DisableSetUnhandledExceptionFilter() {
    try {
        void *addr = (void *)SetUnhandledExceptionFilter;

        if (addr) {
            unsigned char code[16];
            int size = 0;

            code[size++] = 0x33;
            code[size++] = 0xC0;
            code[size++] = 0xC2;
            code[size++] = 0x04;
            code[size++] = 0x00;

            DWORD dwOldFlag, dwTempFlag;
            BOOL result1 =
                VirtualProtect(addr, size, PAGE_EXECUTE_READWRITE, &dwOldFlag);
            BOOL result2 =
                WriteProcessMemory(GetCurrentProcess(), addr, code, size, NULL);
            BOOL result3 = VirtualProtect(addr, size, dwOldFlag, &dwTempFlag);
        }
    } catch (...) {
    }
}

struct premain {
    premain() {
        absl::FailureSignalHandlerOptions failure_signal_handler_options;
        failure_signal_handler_options.symbolize_stacktrace = true;
        failure_signal_handler_options.use_alternate_stack = true;
        failure_signal_handler_options.alarm_on_failure_secs = 5;
        failure_signal_handler_options.call_previous_handler = true;
        absl::InstallFailureSignalHandler(failure_signal_handler_options);

        //DisableSetUnhandledExceptionFilter();
    }
} premain_v;

#endif
std::basic_string<char_t> get_hostfxr_path_wrapper() {
    size_t path_length;
    if (get_hostfxr_path(nullptr, &path_length, nullptr) != 0x80008098)
        throw std::runtime_error("Failed to get hostfxr path.");

    std::basic_string<char_t> hostfxr_path(path_length, '\0');
    if (get_hostfxr_path(hostfxr_path.data(), &path_length, nullptr))
        throw std::runtime_error("Failed to get hostfxr path.");
    return hostfxr_path;
}

c_api_initialize_fn
load_compiler_c_api_initializer(const char *root_assembly_path) {
    auto hostfxr_mod = load_library(get_hostfxr_path_wrapper().c_str());

    hostfxr_handle handle = nullptr;
    std::filesystem::path compiler_path(root_assembly_path);

#ifdef NNCASE_DOTNET_INIT_FOR_CONFIG
    auto hostfxr_initialize =
        (hostfxr_initialize_for_runtime_config_fn)load_symbol(
            hostfxr_mod, "hostfxr_initialize_for_runtime_config");
    auto hr = hostfxr_initialize(
        compiler_path.replace_extension(".runtimeconfig.json").c_str(), nullptr,
        &handle);
#else
    const char_t *args[] = {compiler_path.c_str()};
    auto hostfxr_initialize =
        (hostfxr_initialize_for_dotnet_command_line_fn)load_symbol(
            hostfxr_mod, "hostfxr_initialize_for_dotnet_command_line");
    auto hr = hostfxr_initialize(1, args, nullptr, &handle);
#endif

    THROW_IF_NOT(handle, "Failed to initialize hostfxr: 0x%x.", hr);

    auto hostfxr_get_delegate = (hostfxr_get_runtime_delegate_fn)load_symbol(
        hostfxr_mod, "hostfxr_get_runtime_delegate");

#ifdef NNCASE_DOTNET_INIT_FOR_CONFIG
    load_assembly_and_get_function_pointer_fn hostfxr_get_fn_ptr = nullptr;
    hr =
        hostfxr_get_delegate(handle, hdt_load_assembly_and_get_function_pointer,
                             (void **)&hostfxr_get_fn_ptr);
    THROW_IF_NOT(hostfxr_get_fn_ptr,
                 "Failed to initialize "
                 "hdt_load_assembly_and_get_function_pointer: 0x%x.",
                 hr);
#else
    get_function_pointer_fn hostfxr_get_fn_ptr = nullptr;
    hr = hostfxr_get_delegate(handle, hdt_get_function_pointer,
                              (void **)&hostfxr_get_fn_ptr);
    THROW_IF_NOT(hostfxr_get_fn_ptr,
                 "Failed to initialize hdt_get_function_pointer: 0x%x.", hr);
#endif

    c_api_initialize_fn c_api_initialize = nullptr;
#ifdef NNCASE_DOTNET_INIT_FOR_CONFIG
    hr = hostfxr_get_fn_ptr(compiler_path.c_str(),
                            _T("Nncase.Compiler.Interop.CApi, Nncase.Compiler"),
                            _T("Initialize"), UNMANAGEDCALLERSONLY_METHOD,
                            nullptr, (void **)&c_api_initialize);
#else
    hr = hostfxr_get_fn_ptr(_T("Nncase.Compiler.Interop.CApi, Nncase.Compiler"),
                            _T("Initialize"), UNMANAGEDCALLERSONLY_METHOD,
                            nullptr, nullptr, (void **)&c_api_initialize);
#endif

    THROW_IF_NOT(c_api_initialize,
                 "Failed to initialize Nncase.Compiler.Interop.CApi: 0x%x.",
                 hr);
    return c_api_initialize;
}

nncase_api_mt_t g_nncase_api_mt;
} // namespace

nncase_api_mt_t *nncase_clr_api() { return &g_nncase_api_mt; }

int nncase_clr_initialize(const char *root_assembly_path) {
    if (!g_nncase_api_mt.handle_free) {
        auto init = load_compiler_c_api_initializer(root_assembly_path);
        init(&g_nncase_api_mt);
        g_nncase_api_mt.compiler_initialize();
        // SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
    }

    return 0;
}

int nncase_clr_uninitialize() {
    g_nncase_api_mt = {};
    return 0;
}
