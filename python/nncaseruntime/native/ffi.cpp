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
#include "pystreambuf.h"
#include "pytype_utils.h"
#include "type_casters.h"
#include <iostream>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/version.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <sstream>

namespace py = pybind11;
using namespace nncase;
using namespace nncase::runtime;

namespace {
#ifdef WIN32
#include <Windows.h>
void LaunchDebugger() {
    // Get System directory, typically c:\windows\system32
    std::wstring systemDir(MAX_PATH + 1, '\0');
    UINT nChars = GetSystemDirectoryW(&systemDir[0], systemDir.length());
    if (nChars == 0)
        return;
    systemDir.resize(nChars);

    // Get process ID and create the command line
    DWORD pid = GetCurrentProcessId();
    std::wostringstream s;
    s << systemDir << L"\\vsjitdebugger.exe -p " << pid;
    std::wstring cmdLine = s.str();

    // Start debugger process
    STARTUPINFOW si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcessW(NULL, &cmdLine[0], NULL, NULL, FALSE, 0, NULL, NULL,
                        &si, &pi))
        return;

    // Close debugger process handles to eliminate resource leak
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    // Wait for the debugger to attach
    while (!IsDebuggerPresent())
        Sleep(100);
}
#endif
} // namespace

PYBIND11_MODULE(_nncaseruntime, m) {
    m.doc() = "nncase runtime Library";
    m.attr("__version__") = NNCASE_VERSION;

    // LaunchDebugger();

#include "runtime_tensor.inl"

    py::class_<interpreter>(m, "Interpreter")
        .def(py::init())
        .def("load_model",
             [](interpreter &interp, gsl::span<const gsl::byte> buffer) {
                 interp.load_model(buffer, true).unwrap_or_throw();
             })
        .def_property_readonly("inputs_size", &interpreter::inputs_size)
        .def_property_readonly("outputs_size", &interpreter::outputs_size)
        .def("get_input_desc", &interpreter::input_desc)
        .def("get_output_desc", &interpreter::output_desc)
        .def("get_input_tensor",
             [](interpreter &interp, size_t index) {
                 return interp.input_tensor(index).unwrap_or_throw();
             })
        .def("set_input_tensor",
             [](interpreter &interp, size_t index, runtime_tensor tensor) {
                 return interp.input_tensor(index, tensor).unwrap_or_throw();
             })
        .def("get_output_tensor",
             [](interpreter &interp, size_t index) {
                 return interp.output_tensor(index).unwrap_or_throw();
             })
        .def("set_output_tensor",
             [](interpreter &interp, size_t index, runtime_tensor tensor) {
                 return interp.output_tensor(index, tensor).unwrap_or_throw();
             })
        .def("set_profiling",
             [](interpreter &interp, uint8_t enabled) {
                 interp.set_profiling(enabled);
             })
        .def("run",
             [](interpreter &interp) { interp.run().unwrap_or_throw(); });
}
