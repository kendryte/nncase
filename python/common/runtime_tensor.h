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
#include "nncase/ntt/utility.h"
#include "nncase/object.h"
#include "pytype_utils.h"
#include "type_casters.h"
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

using namespace nncase::runtime;
namespace py = pybind11;

namespace nncase {
inline void register_runtime_tensor(py::module &m) {
    py::class_<tensor_desc>(m, "TensorDesc")
        .def_property(
            "dtype",
            [](const tensor_desc &desc) { return to_dtype(desc.datatype); },
            [](tensor_desc &desc, py::object dtype) {
                desc.datatype = from_dtype(py::dtype::from_args(dtype));
            })
        .def_readwrite("start", &tensor_desc::start)
        .def_readwrite("size", &tensor_desc::size);

    py::class_<runtime_tensor>(m, "RuntimeTensor")
        // .def_static(
        //     "from_object",
        //     [](object obj) {
        //         auto ref_type = nncase::reference_type_t(
        //             std::in_place, datatype_t::attention_kv_cache);
        //         auto tensor =
        //             nncase::runtime::detail::create(
        //                 ref_type, {},
        //                 nncase::runtime::hrt::memory_pool_t::pool_cpu_only)
        //                 .unwrap_or_throw();
        //         auto host_buffer =
        //         tensor->buffer().as_host().unwrap_or_throw(); auto
        //         mapped_data = host_buffer.map(map_write).unwrap_or_throw();
        //         auto dest_buffer =
        //         ntt::span_cast<object>(mapped_data.buffer()); dest_buffer[0]
        //         = obj; return runtime_tensor(tensor);
        //     })
        .def_static(
            "from_numpy",
            [](py::array arr) {
                arr = py::array::ensure(arr, py::array::c_style);
                auto src_buffer = arr.request();
                auto datatype = from_dtype(arr);
                auto tensor =
                    host_runtime_tensor::create(
                        datatype, to_rt_shape(src_buffer.shape),
                        to_rt_strides(src_buffer.itemsize, src_buffer.strides),
                        std::span(reinterpret_cast<std::byte *>(src_buffer.ptr),
                                  src_buffer.size * src_buffer.itemsize), 
                                  true, hrt::pool_shared)
                        .unwrap_or_throw();
                // arr.inc_ref();
                return tensor;
            })
        .def("copy_to",
             [](runtime_tensor &from, runtime_tensor &to) {
                 from.copy_to(to).unwrap_or_throw();
             })
        .def("to_numpy",
             [](runtime_tensor &tensor) {
                 auto host = tensor.to_host().unwrap_or_throw();
                 auto src_map = std::move(
                     hrt::map(host, runtime::map_read).unwrap_or_throw());
                 auto src_buffer = src_map.buffer();
                 return py::array(
                     to_dtype(tensor.datatype()), tensor.shape(),
                     to_py_strides(runtime::get_bytes(tensor.datatype()),
                                   tensor.strides()),
                     src_buffer.data());
             })
        .def_property_readonly(
            "dtype",
            [](runtime_tensor &tensor) { return to_dtype(tensor.datatype()); })
        .def_property_readonly("shape", [](runtime_tensor &tensor) {
            return to_py_shape(tensor.shape());
        });
}
} // namespace nncase
