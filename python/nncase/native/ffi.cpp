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
    m.def("launch_debugger", []() { nncase_clr_api()->luanch_debugger(); });

#include "runtime_tensor.inl"

    py::enum_<nncase_model_quant_mode_t>(m, "ModelQuantMode")
        .value("NoQuant", nncase_mqm_no_quant)
        .value("UsePTQ", nncase_mqm_use_ptq)
        .value("UseQAT", nncase_mqm_use_qat);

    py::enum_<nncase_dump_flags_t>(m, "DumpFlags", py::arithmetic())
        .value("Nothing", nncase_dump_flags_none)
        .value("ImportOps", nncase_dump_flags_import_ops)
        .value("PassIR", nncase_dump_flags_pass_ir)
        .value("EGraphCost", nncase_dump_flags_egraph_cost)
        .value("Rewrite", nncase_dump_flags_rewrite)
        .value("Calibration", nncase_dump_flags_calibration)
        .value("Evaluator", nncase_dump_flags_evaluator)
        .value("Compile", nncase_dump_flags_compile)
        .value("Tiling", nncase_dump_flags_tiling)
        .value("Schedule", nncase_dump_flags_schedule)
        .value("CodeGen", nncase_dump_flags_codegen);

    py::enum_<nncase_calib_method_t>(m, "CalibMethod")
        .value("NoClip", nncase_calib_noclip)
        .value("Kld", nncase_calib_kld);

    py::enum_<nncase_quant_type_t>(m, "QuantType")
        .value("Uint8", nncase_qt_uint8)
        .value("Int8", nncase_qt_int8)
        .value("Int16", nncase_qt_int16);

    py::enum_<nncase_input_type_t>(m, "InputType")
        .value("Uint8", nncase_it_uint8)
        .value("Int8", nncase_it_int8)
        .value("Float32", nncase_it_float32);

    py::enum_<nncase_finetune_weights_method_t>(m, "FineTuneWeightsMethod")
        .value("NoFineTuneWeights", nncase_no_finetune_weights)
        .value("UseSquant", nncase_finetune_weights_squant)
        .value("UseAdaRound", nncase_finetune_weights_adaround);

    py::class_<compile_options>(m, "CompileOptions")
        .def(py::init())
        .def_property(
            "input_format", py::overload_cast<>(&compile_options::input_format),
            py::overload_cast<std::string_view>(&compile_options::input_format))
        .def_property(
            "dump_dir", py::overload_cast<>(&compile_options::dump_dir),
            py::overload_cast<std::string_view>(&compile_options::dump_dir))
        .def_property("dump_flags",
                      py::overload_cast<>(&compile_options::dump_flags),
                      py::overload_cast<nncase_dump_flags_t>(
                          &compile_options::dump_flags))
        .def_property("quantize_options",
                      py::overload_cast<>(&compile_options::quantize_options),
                      py::overload_cast<const quantize_options &>(
                          &compile_options::quantize_options))
        .def_property("preprocess",
                      py::overload_cast<>(&compile_options::preprocess),
                      py::overload_cast<bool>(&compile_options::preprocess))
        .def_property(
            "input_layout", py::overload_cast<>(&compile_options::input_layout),
            py::overload_cast<std::string_view>(&compile_options::input_layout))
        .def_property("output_layout",
                      py::overload_cast<>(&compile_options::output_layout),
                      py::overload_cast<std::string_view>(
                          &compile_options::output_layout))
        .def_property(
            "input_file", py::overload_cast<>(&compile_options::input_file),
            py::overload_cast<std::string_view>(&compile_options::input_file))
        .def_property("input_type",
                      py::overload_cast<>(&compile_options::input_type),
                      py::overload_cast<nncase_input_type_t>(
                          &compile_options::input_type))
        .def_property(
            "input_shape", py::overload_cast<>(&compile_options::input_shape),
            py::overload_cast<std::string_view>(&compile_options::input_shape))
        .def_property(
            "input_range", py::overload_cast<>(&compile_options::input_range),
            py::overload_cast<std::string_view>(&compile_options::input_range))
        .def_property("swapRB", py::overload_cast<>(&compile_options::swapRB),
                      py::overload_cast<bool>(&compile_options::swapRB))
        .def_property(
            "letterbox_value",
            py::overload_cast<>(&compile_options::letterbox_value),
            py::overload_cast<float>(&compile_options::letterbox_value))
        .def_property(
            "mean", py::overload_cast<>(&compile_options::mean),
            py::overload_cast<std::string_view>(&compile_options::mean))
        .def_property(
            "std", py::overload_cast<>(&compile_options::std),
            py::overload_cast<std::string_view>(&compile_options::std))
        .def_property(
            "shape_bucket_options",
            py::overload_cast<>(&compile_options::shape_bucket_options),
            py::overload_cast<const shape_bucket_options &>(
                &compile_options::shape_bucket_options));

    py::class_<target>(m, "Target")
        .def(py::init<std::string_view>())
        .def_static("exists", &target::exists);

    py::class_<quantize_options>(m, "QuantizeOptions")
        .def(py::init())
        .def_property(
            "calibration_dataset",
            py::overload_cast<>(&quantize_options::calibration_dataset),
            py::overload_cast<const calibration_dataset_provider &>(
                &quantize_options::calibration_dataset))
        .def_property("model_quant_mode",
                      py::overload_cast<>(&quantize_options::model_quant_mode),
                      py::overload_cast<nncase_model_quant_mode_t>(
                          &quantize_options::model_quant_mode))
        .def_property("calibrate_method",
                      py::overload_cast<>(&quantize_options::calibrate_method),
                      py::overload_cast<nncase_calib_method_t>(
                          &quantize_options::calibrate_method))
        .def_property("quant_type",
                      py::overload_cast<>(&quantize_options::quant_type),
                      py::overload_cast<nncase_quant_type_t>(
                          &quantize_options::quant_type))
        .def_property("w_quant_type",
                      py::overload_cast<>(&quantize_options::w_quant_type),
                      py::overload_cast<nncase_quant_type_t>(
                          &quantize_options::w_quant_type))
        .def_property(
            "finetune_weights_method",
            py::overload_cast<>(&quantize_options::finetune_weights_method),
            py::overload_cast<nncase_finetune_weights_method_t>(
                &quantize_options::finetune_weights_method))
        .def_property("use_mix_quant",
                      py::overload_cast<>(&quantize_options::use_mix_quant),
                      py::overload_cast<bool>(&quantize_options::use_mix_quant))
        .def_property("quant_scheme",
                      py::overload_cast<>(&quantize_options::quant_scheme),
                      py::overload_cast<std::string_view>(
                          &quantize_options::quant_scheme))
        .def_property(
            "quant_scheme_strict_mode",
            py::overload_cast<>(&quantize_options::quant_scheme_strict_mode),
            py::overload_cast<bool>(
                &quantize_options::quant_scheme_strict_mode))
        .def_property(
            "export_quant_scheme",
            py::overload_cast<>(&quantize_options::export_quant_scheme),
            py::overload_cast<bool>(&quantize_options::export_quant_scheme))
        .def_property("export_weight_range_by_channel",
                      py::overload_cast<>(
                          &quantize_options::export_weight_range_by_channel),
                      py::overload_cast<bool>(
                          &quantize_options::export_weight_range_by_channel))
        .def_property(
            "dump_quant_error",
            py::overload_cast<>(&quantize_options::dump_quant_error),
            py::overload_cast<bool>(&quantize_options::dump_quant_error))
        .def_property(
            "dump_quant_error_symmetric_for_signed",
            py::overload_cast<>(
                &quantize_options::dump_quant_error_symmetric_for_signed),
            py::overload_cast<bool>(
                &quantize_options::dump_quant_error_symmetric_for_signed));

    py::class_<shape_bucket_options>(m, "ShapeBucketOptions")
        .def(py::init())
        .def_property("enable",
                      py::overload_cast<>(&shape_bucket_options::enable),
                      py::overload_cast<bool>(&shape_bucket_options::enable))
        .def_property(
            "range_info",
            py::overload_cast<>(&shape_bucket_options::range_info),
            py::overload_cast<std::map<std::string, std::tuple<int, int>>>(
                &shape_bucket_options::range_info))
        .def_property(
            "segments_count",
            py::overload_cast<>(&shape_bucket_options::segments_count),
            py::overload_cast<int>(&shape_bucket_options::segments_count))
        .def_property("fix_var_map",
                      py::overload_cast<>(&shape_bucket_options::fix_var_map),
                      py::overload_cast<std::map<std::string, int>>(
                          &shape_bucket_options::fix_var_map));

    py::class_<calibration_dataset_provider>(m, "CalibrationDatasetProvider")
        .def(py::init([](py::list dataset, size_t samples_count,
                         py::list fn_params) {
            std::vector<clr_object_handle_t> dataset_handles(dataset.size());
            std::vector<clr_object_handle_t> param_handles(fn_params.size());
            for (size_t i = 0; i < dataset_handles.size(); i++) {
                dataset_handles[i] = dataset[i].cast<rtvalue &>().get();
            }
            for (size_t i = 0; i < param_handles.size(); i++) {
                param_handles[i] = fn_params[i].cast<var &>().get();
            }

            array dataset_arr(nncase_array_rtvalue, dataset_handles.data(),
                              dataset.size());
            array fn_params_arr(nncase_array_var, param_handles.data(),
                                fn_params.size());
            return calibration_dataset_provider(std::move(dataset_arr),
                                                samples_count,
                                                std::move(fn_params_arr));
        }));

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

    py::class_<expr>(m, "Expr").def("evaluate", [](expr &expr, py::list params,
                                                   py::list inputs) {
        std::vector<clr_object_handle_t> param_handles(params.size());
        std::vector<clr_object_handle_t> input_handles(inputs.size());
        for (size_t i = 0; i < param_handles.size(); i++) {
            param_handles[i] = params[i].cast<var &>().get();
        }
        for (size_t i = 0; i < input_handles.size(); i++) {
            input_handles[i] = inputs[i].cast<rtvalue &>().get();
        }

        array params_arr(nncase_array_var, param_handles.data(), inputs.size());
        array inputs_arr(nncase_array_rtvalue, input_handles.data(),
                         inputs.size());
        return expr.evaluate(params_arr, inputs_arr);
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
        .def("import_tflite_module", &compiler::import_tflite_module)
        .def("import_onnx_module", &compiler::import_onnx_module)
        .def("import_ncnn_module", &compiler::import_ncnn_module)
        .def("compile", &compiler::compile)
        .def("gencode", &compiler::gencode);

    py::class_<compile_session>(m, "CompileSession")
        .def(py::init<const target &, const compile_options &>())
        .def_property_readonly("compiler", &compile_session::compiler);

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
        .def("get_input_shape",
             [](interpreter &interp, size_t index) {
                 return to_py_shape(interp.input_shape(index));
             })
        .def("get_output_shape",
             [](interpreter &interp, size_t index) {
                 return to_py_shape(interp.output_shape(index));
             })
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
