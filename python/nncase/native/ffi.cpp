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
#include <nncase/compiler.h>
#include <nncase/ir/debug.h>
#include <nncase/ir/evaluator.h>
#include <nncase/ir/graph.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/schedule/scheduler.h>
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

namespace
{
#ifdef WIN32
#include <Windows.h>
void LaunchDebugger()
{
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

    if (!CreateProcessW(NULL, &cmdLine[0], NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi))
        return;

    // Close debugger process handles to eliminate resource leak
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    // Wait for the debugger to attach
    while (!IsDebuggerPresent())
        Sleep(100);
}
#endif

schedule::model_schedule_result schedule(target &target, ir::graph &graph)
{
    schedule::scheduler sched(target, graph, graph.outputs());
    return sched.schedule(true);
}

runtime_tensor eval_tensor_to_rt_tensor(const ir::evaluate_tensor &src)
{
    return hrt::create(src.datatype(), src.shape(), src.strides(), src.buffer(), false).unwrap_or_throw();
}

class graph_evaluator
{
public:
    graph_evaluator(target &target, ir::graph &graph)
        : graph_(graph), schedule_result_(schedule(target, graph)), evaluator_(schedule_result_)
    {
    }

    size_t outputs_size() const noexcept
    {
        return graph_.outputs().size();
    }

    runtime::runtime_tensor input_at(size_t index)
    {
        return eval_tensor_to_rt_tensor(evaluator_.input_at(index));
    }

    runtime::runtime_tensor output_at(size_t index)
    {
        return eval_tensor_to_rt_tensor(evaluator_.output_at(index));
    }

    void run()
    {
        evaluator_.evaluate();
    }

private:
    ir::graph &graph_;
    schedule::model_schedule_result schedule_result_;
    ir::evaluator evaluator_;
};
}

PYBIND11_MODULE(_nncase, m)
{
    m.doc() = "nncase Library";
    m.attr("__version__") = NNCASE_VERSION;

    // LaunchDebugger();
    py::class_<std::filesystem::path>(m, "Path")
        .def(py::init<std::string>());
    py::implicitly_convertible<std::string, std::filesystem::path>();

    py::class_<compile_options>(m, "CompileOptions")
        .def(py::init())
        .def_readwrite("dump_ir", &compile_options::dump_ir)
        .def_readwrite("dump_asm", &compile_options::dump_asm)
        .def_readwrite("target", &compile_options::target)
        .def_readwrite("dump_dir", &compile_options::dump_dir)
        .def_readwrite("is_fpga", &compile_options::is_fpga)
        .def_readwrite("input_type", &compile_options::input_type)
        .def_readwrite("output_type", &compile_options::output_type)
        .def_readwrite("quant_type", &compile_options::quant_type)
        .def_readwrite("image_format", &compile_options::image_format)
        .def_readwrite("mean", &compile_options::mean)
        .def_readwrite("scale", &compile_options::scale)
        .def_readwrite("input_range", &compile_options::input_range)
        .def_readwrite("input_shape", &compile_options::input_shape)
        .def_readwrite("w_quant_type", &compile_options::w_quant_type)
        .def_readwrite("benchmark_only", &compile_options::benchmark_only);

    py::class_<import_options>(m, "ImportOptions")
        .def(py::init())
        .def_readwrite("input_layout", &import_options::input_layout)
        .def_readwrite("output_layout", &import_options::output_layout)
        .def_readwrite("output_arrays", &import_options::output_arrays);

    py::class_<ptq_tensor_options>(m, "PTQTensorOptions")
        .def(py::init())
        .def_readwrite("calibrate_method", &ptq_tensor_options::calibrate_method)
        .def("set_tensor_data", [](ptq_tensor_options &o, py::bytes bytes) {
            uint8_t *buffer;
            py::ssize_t length;
            if (PyBytes_AsStringAndSize(bytes.ptr(), reinterpret_cast<char **>(&buffer), &length))
                throw std::invalid_argument("Invalid bytes");
            o.tensor_data.assign(buffer, buffer + length);
        })
        .def_readwrite("samples_count", &ptq_tensor_options::samples_count)
        .def_readwrite("input_mean", &ptq_tensor_options::input_mean)
        .def_readwrite("input_std", &ptq_tensor_options::input_std);

    py::class_<graph_evaluator>(m, "GraphEvaluator")
        .def_property_readonly("outputs_size", &graph_evaluator::outputs_size)
        .def("get_input_tensor", &graph_evaluator::input_at)
        .def("get_output_tensor", &graph_evaluator::output_at)
        .def("run", &graph_evaluator::run);

    py::class_<compiler>(m, "Compiler")
        .def(py::init(&compiler::create))
        .def("import_tflite", &compiler::import_tflite)
        .def("import_onnx", &compiler::import_onnx)
        .def("import_caffe", &compiler::import_caffe)
        .def("compile", &compiler::compile)
        .def("use_ptq", py::overload_cast<ptq_tensor_options>(&compiler::use_ptq))
        .def("gencode", [](compiler &c, std::ostream &stream) { c.gencode(stream); })
        .def("gencode_tobytes", [](compiler &c) {
            std::stringstream ss;
            c.gencode(ss);
            return py::bytes(ss.str());
        })
        .def("create_evaluator", [](compiler &c, uint32_t stage) {
            auto &graph = c.graph(stage);
            return std::make_unique<graph_evaluator>(c.target(), graph);
        });

#include "runtime_tensor.inl"

    py::class_<interpreter>(m, "Simulator")
        .def(py::init())
        .def("load_model", [](interpreter &interp, gsl::span<const gsl::byte> buffer) { interp.load_model(buffer).unwrap_or_throw(); })
        .def_property_readonly("inputs_size", &interpreter::inputs_size)
        .def_property_readonly("outputs_size", &interpreter::outputs_size)
        .def("get_input_desc", &interpreter::input_desc)
        .def("get_output_desc", &interpreter::output_desc)
        .def("get_input_tensor", [](interpreter &interp, size_t index) { return interp.input_tensor(index).unwrap_or_throw(); })
        .def("set_input_tensor", [](interpreter &interp, size_t index, runtime_tensor tensor) { return interp.input_tensor(index, tensor).unwrap_or_throw(); })
        .def("get_output_tensor", [](interpreter &interp, size_t index) { return interp.output_tensor(index).unwrap_or_throw(); })
        .def("set_output_tensor", [](interpreter &interp, size_t index, runtime_tensor tensor) { return interp.output_tensor(index, tensor).unwrap_or_throw(); })
        .def("run", [](interpreter &interp) { interp.run().unwrap_or_throw(); });

    m.def("test_target", [](std::string name) {
        try
        {
            auto target = plugin_loader::create_target(name);
            return true;
        }
        catch (...)
        {
            return false;
        }
    });
}
