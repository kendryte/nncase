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
#pragma once
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/debug.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace nncase {
pybind11::dtype to_dtype(typecode_t type) {
    namespace py = pybind11;

    switch (type) {
    case dt_boolean:
        return py::dtype::of<bool>();
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
    case dt_float16:
        return py::dtype("float16");
    case dt_float32:
        return py::dtype::of<float>();
    case dt_float64:
        return py::dtype::of<double>();
    default:
        throw std::runtime_error("Unsupported dtype " + to_string(type));
    }
}

pybind11::dtype to_dtype(const datatype_t type) {
    auto primtype = type.as<prim_type_t>();
    if (primtype.is_err()) {
        throw std::runtime_error("Only support primtype.");
    }
    return to_dtype(primtype.unwrap()->typecode());
}

typecode_t from_dtype(pybind11::dtype dtype) {
    namespace py = pybind11;

    if (dtype.is(py::dtype::of<bool>()))
        return dt_boolean;
    else if (dtype.is(py::dtype::of<uint8_t>()))
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
    else if (dtype.is(py::dtype("float16")))
        return dt_float16;
    else if (dtype.is(py::dtype::of<float>()))
        return dt_float32;
    else if (dtype.is(py::dtype::of<double>()))
        return dt_float64;
    throw std::runtime_error("Unsupported dtype " +
                             (std::string)py::str(dtype));
}

dims_t to_rt_shape(const std::vector<pybind11::ssize_t> &value) {
    dims_t shape(value.size());
    for (size_t i = 0; i < shape.size(); i++)
        shape[i] = (size_t)value[i];
    return shape;
}

strides_t to_rt_strides(size_t elemsize,
                        const std::vector<pybind11::ssize_t> &value) {
    strides_t strides(value.size());
    for (size_t i = 0; i < strides.size(); i++)
        strides[i] = (size_t)value[i] / elemsize;
    return strides;
}

std::vector<pybind11::ssize_t> to_py_shape(gsl::span<const size_t> value) {
    namespace py = pybind11;

    std::vector<py::ssize_t> shape(value.size());
    for (size_t i = 0; i < shape.size(); i++)
        shape[i] = (py::ssize_t)value[i];
    return shape;
}

std::vector<pybind11::ssize_t> to_py_strides(size_t elemsize,
                                             gsl::span<const size_t> value) {
    namespace py = pybind11;

    std::vector<py::ssize_t> strides(value.size());
    for (size_t i = 0; i < strides.size(); i++)
        strides[i] = (py::ssize_t)value[i] * elemsize;
    return strides;
}
} // namespace nncase
