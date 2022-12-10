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
using namespace nncase::clr;
using namespace nncase::runtime;

namespace {} // namespace

namespace pybind11::detail {
std::atomic_bool g_python_shutdown = false;
}

PYBIND11_MODULE(_nncase, m) {
    m.doc() = "nncase Library";
    m.attr("__version__") = NNCASE_VERSION NNCASE_VERSION_SUFFIX;

    m.add_object("_cleanup", py::capsule([]() {
                     nncase_clr_uninitialize();
                     pybind11::detail::g_python_shutdown.store(
                         true, std::memory_order_release);
                 }));
    m.def("initialize", nncase_clr_initialize);
    m.def("launch_debugger", nncase_clr_launch_debugger);
    m.def("target_exists", [](std::string_view target_name) {
        return nncase_clr_target_exists(target_name.data(),
                                        target_name.length());
    });

#include "runtime_tensor.inl"

    py::class_<compile_options>(m, "CompileOptions")
        .def(py::init())
        .def_property(
            "input_format", py::overload_cast<>(&compile_options::input_format),
            py::overload_cast<std::string_view>(&compile_options::input_format))
        .def_property(
            "target", py::overload_cast<>(&compile_options::target),
            py::overload_cast<std::string_view>(&compile_options::target))
        .def_property("dump_level",
                      py::overload_cast<>(&compile_options::dump_level),
                      py::overload_cast<int32_t>(&compile_options::dump_level))
        .def_property(
            "dump_dir", py::overload_cast<>(&compile_options::dump_dir),
            py::overload_cast<std::string_view>(&compile_options::dump_dir));

    py::class_<rtvalue>(m, "RTValue")
        .def_static(
            "from_runtime_tensor",
            [](const runtime_tensor &tensor) { return rtvalue(tensor.impl()); })
        .def("to_runtime_tensor",
             [](const rtvalue &value) {
                 return runtime_tensor(
                     value.to_value().as<tensor>().unwrap_or_throw());
             })
        .def("to_runtime_tensors", [](const rtvalue &value) {
            auto v = value.to_value();
            if (v.is_a<tensor>()) {
                return std::vector<runtime_tensor>{
                    runtime_tensor(v.as<tensor>().unwrap_or_throw())};
            } else {
                auto t = v.as<tuple>().unwrap_or_throw();
                std::vector<runtime_tensor> tensors(t->fields().size());
                for (size_t i = 0; i < tensors.size(); i++) {
                    tensors[i] = runtime_tensor(
                        t->fields()[i].as<tensor>().unwrap_or_throw());
                }
                return tensors;
            }
        });

    py::class_<expr>(m, "Expr").def("evaluate", [](expr &expr, py::list inputs,
                                                   py::list params) {
        std::vector<clr_object_handle_t> input_handles(inputs.size());
        std::vector<clr_object_handle_t> param_handles(params.size());
        for (size_t i = 0; i < input_handles.size(); i++) {
            input_handles[i] = inputs[i].cast<rtvalue &>().get();
        }
        for (size_t i = 0; i < param_handles.size(); i++) {
            param_handles[i] = params[i].cast<var &>().get();
        }

        array inputs_arr(nncase_array_rtvalue, input_handles.data(),
                         inputs.size());
        array params_arr(nncase_array_var, param_handles.data(), inputs.size());
        return expr.evaluate(inputs_arr, params_arr);
    });

    py::class_<var, expr>(m, "Var");

    py::class_<function, expr>(m, "Function")
        .def_property_readonly("body", &function::body)
        .def_property_readonly("parameters", [](function &function) {
            return function.parameters().to_vector<var>();
        });

    py::class_<ir_module>(m, "IRModule")
        .def_property_readonly("entry", &ir_module::entry);

    py::class_<compiler>(m, "Compiler")
        .def(py::init<const compile_options &>())
        .def("import_module", &compiler::import_module)
        .def("compile", &compiler::compile)
        .def("gencode", &compiler::gencode);

    py::class_<interpreter>(m, "Simulator")
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
        .def("run",
             [](interpreter &interp) { interp.run().unwrap_or_throw(); });
}
