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

PYBIND11_MODULE(_knn, m)
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

    py::class_<interpreter>(m, "Simulator")
        .def("load_model", [](interpreter &interp, gsl::span<const gsl::byte> buffer) {
            interp.load_model(buffer).unwrap_or_throw();
        });
}
