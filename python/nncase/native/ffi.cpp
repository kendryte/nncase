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
#include "pystreambuf.h"
#include <iostream>
#include <nncase/compiler.h>
#include <nncase/ir/debug.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/version.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <sstream>

namespace py = pybind11;
using namespace nncase;
using namespace nncase::runtime;

namespace pybind11
{
namespace detail
{
    template <>
    struct type_caster<std::span<const uint8_t>>
    {
    public:
        PYBIND11_TYPE_CASTER(std::span<const uint8_t>, _("bytes"));

        bool load(handle src, bool)
        {
            if (!py::isinstance<py::bytes>(src))
                return false;

            uint8_t *buffer;
            py::ssize_t length;
            if (PyBytes_AsStringAndSize(src.ptr(), reinterpret_cast<char **>(&buffer), &length))
                return false;
            value = { buffer, (size_t)length };
            return true;
        }
    };

    template <>
    struct type_caster<gsl::span<const gsl::byte>>
    {
    public:
        PYBIND11_TYPE_CASTER(gsl::span<const gsl::byte>, _("bytes"));

        bool load(handle src, bool)
        {
            if (!py::isinstance<py::bytes>(src))
                return false;

            uint8_t *buffer;
            py::ssize_t length;
            if (PyBytes_AsStringAndSize(src.ptr(), reinterpret_cast<char **>(&buffer), &length))
                return false;
            value = { (const gsl::byte *)buffer, (size_t)length };
            return true;
        }
    };
}
}

namespace
{
py::dtype to_dtype(datatype_t type)
{
    switch (type)
    {
    case dt_uint8:
        return py::dtype::of<uint8_t>();
    case dt_uint16:
        return py::dtype::of<uint16_t>();
    case dt_uint32:
        return py::dtype::of<uint32_t>();
    case dt_uint64:
        return py::dtype::of<uint64_t>();
    case dt_int8:
        return py::dtype::of<int8_t>();
    case dt_int16:
        return py::dtype::of<int16_t>();
    case dt_int32:
        return py::dtype::of<int32_t>();
    case dt_int64:
        return py::dtype::of<int64_t>();
    case dt_float32:
        return py::dtype::of<float>();
    case dt_float64:
        return py::dtype::of<double>();
    default:
        throw std::runtime_error("Unsupported dtype " + (std::string)datatype_names(type));
    }
}

datatype_t from_dtype(py::dtype dtype)
{
    if (dtype.is(py::dtype::of<uint8_t>()))
        return dt_uint8;
    else if (dtype.is(py::dtype::of<uint16_t>()))
        return dt_uint16;
    else if (dtype.is(py::dtype::of<uint32_t>()))
        return dt_uint32;
    else if (dtype.is(py::dtype::of<uint64_t>()))
        return dt_uint64;
    else if (dtype.is(py::dtype::of<int8_t>()))
        return dt_int8;
    else if (dtype.is(py::dtype::of<int16_t>()))
        return dt_int16;
    else if (dtype.is(py::dtype::of<int32_t>()))
        return dt_int32;
    else if (dtype.is(py::dtype::of<int64_t>()))
        return dt_int64;
    else if (dtype.is(py::dtype::of<float>()))
        return dt_float32;
    else if (dtype.is(py::dtype::of<double>()))
        return dt_float64;
    throw std::runtime_error("Unsupported dtype " + (std::string)py::str(dtype));
}

runtime_shape_t to_rt_shape(const std::vector<pybind11::ssize_t> &value)
{
    runtime_shape_t shape(value.size());
    for (size_t i = 0; i < shape.size(); i++)
        shape[i] = (size_t)value[i];
    return shape;
}

std::vector<py::ssize_t> to_py_shape(const runtime_shape_t &value)
{
    std::vector<py::ssize_t> shape(value.size());
    for (size_t i = 0; i < shape.size(); i++)
        shape[i] = (py::ssize_t)value[i];
    return shape;
}
}

PYBIND11_MODULE(_nncase, m)
{
    m.doc() = "NNCase Library";
    m.attr("__version__") = NNCASE_VERSION;

    py::class_<std::filesystem::path>(m, "Path")
        .def(py::init<std::string>());
    py::implicitly_convertible<std::string, std::filesystem::path>();

    py::class_<compile_options>(m, "CompileOptions")
        .def(py::init())
        .def_readwrite("dump_ir", &compile_options::dump_ir)
        .def_readwrite("dump_asm", &compile_options::dump_asm)
        .def_readwrite("target", &compile_options::target)
        .def_readwrite("dump_dir", &compile_options::dump_dir);

    py::class_<import_options>(m, "ImportOptions")
        .def(py::init())
        .def_readwrite("output_arrays", &import_options::output_arrays);

    py::class_<compiler>(m, "Compiler")
        .def(py::init(&compiler::create))
        .def("import_tflite", &compiler::import_tflite)
        .def("compile", &compiler::compile)
        .def("gencode", [](compiler &c, std::ostream &stream) {
            c.gencode(stream);
        })
        .def("gencode_tobytes", [](compiler &c) {
            std::stringstream ss;
            c.gencode(ss);
            return py::bytes(ss.str());
        });

    py::enum_<memory_location_t>(m, "MemoryLocation")
        .value("Input", mem_input)
        .value("Output", mem_output)
        .value("Data", mem_data)
        .value("Rdata", mem_rdata);

    py::class_<memory_range>(m, "MemoryRange")
        .def_readwrite("location", &memory_range::memory_location)
        .def_property(
            "dtype", [](const memory_range &range) { return to_dtype(range.datatype); },
            [](memory_range &range, py::object dtype) { range.datatype = from_dtype(py::dtype::from_args(dtype)); })
        .def_readwrite("start", &memory_range::start)
        .def_readwrite("size", &memory_range::size);

    py::class_<runtime_tensor>(m, "RuntimeTensor")
        .def("from_numpy", [](py::array arr) {
            auto src_buffer = arr.request();
            auto tensor = host_runtime_tensor::create(
                from_dtype(arr.dtype()),
                to_rt_shape(src_buffer.shape),
                gsl::make_span(reinterpret_cast<gsl::byte *>(src_buffer.ptr), src_buffer.size * src_buffer.itemsize),
                [=](gsl::span<gsl::byte>) { arr.dec_ref(); })
                              .unwrap_or_throw();
            arr.inc_ref();
            return tensor;
        })
        .def("to_numpy", [](runtime_tensor &tensor) {
            auto host = tensor.as_host().unwrap_or_throw();
            auto src_buffer = host_runtime_tensor::buffer(host);
            return py::array(
                to_dtype(tensor.datatype()),
                tensor.shape(),
                src_buffer.data());
        })
        .def_property_readonly("dtype", [](runtime_tensor &tensor) {
            return to_dtype(tensor.datatype());
        })
        .def_property_readonly("shape", [](runtime_tensor &tensor) {
            return to_py_shape(tensor.shape());
        });

    py::class_<interpreter>(m, "Simulator")
        .def(py::init())
        .def("load_model", [](interpreter &interp, gsl::span<const gsl::byte> buffer) {
            interp.load_model(buffer).unwrap_or_throw();
        })
        .def_property_readonly("inputs_size", &interpreter::inputs_size)
        .def_property_readonly("outputs_size", &interpreter::outputs_size)
        .def("get_input_desc", &interpreter::input_desc)
        .def("get_output_desc", &interpreter::input_desc)
        .def("get_input_tensor", [](interpreter &interp, size_t index) {
            return interp.input_tensor(index).unwrap_or_throw();
        })
        .def("set_input_tensor", [](interpreter &interp, size_t index, runtime_tensor tensor) {
            return interp.input_tensor(index, tensor).unwrap_or_throw();
        })
        .def("get_output_tensor", [](interpreter &interp, size_t index) {
            return interp.output_tensor(index).unwrap_or_throw();
        })
        .def("set_output_tensor", [](interpreter &interp, size_t index, runtime_tensor tensor) {
            return interp.output_tensor(index, tensor).unwrap_or_throw();
        })
        .def("run", [](interpreter &interp) {
            interp.run().unwrap_or_throw();
        });
}
