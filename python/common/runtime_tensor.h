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
#include "nncase/runtime/util.h"
#include "pytype_utils.h"
#include "type_casters.h"
#include <nncase/llm/paged_attention_kv_cache.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

using namespace nncase::runtime;
namespace py = pybind11;

namespace nncase {
inline py::class_<runtime_tensor> register_runtime_tensor(py::module &m) {
    py::class_<tensor_desc>(m, "TensorDesc")
        .def_property(
            "dtype",
            [](const tensor_desc &desc) { return to_dtype(desc.datatype); },
            [](tensor_desc &desc, py::object dtype) {
                desc.datatype = from_dtype(py::dtype::from_args(dtype));
            })
        .def_readwrite("start", &tensor_desc::start)
        .def_readwrite("size", &tensor_desc::size);

    auto rt_class =
        py::class_<runtime_tensor>(m, "RuntimeTensor")
            .def_static("from_object",
                        [](const object &obj) {
                            if (!obj.is_a<llm::paged_attention_kv_cache>()) {
                                throw std::invalid_argument(
                                    "currently only support objects type "
                                    "llm::paged_attention_kv_cache");
                            }

                            auto ref_type = nncase::reference_type_t(
                                std::in_place,
                                datatype_t::paged_attention_kv_cache);

                            auto object_ptrs = new object_node *[1];
                            auto bytes_span = as_span<std::byte>(
                                std::span<object_node *>(object_ptrs, 1));
                            hrt::data_deleter_t deleter = [](std::byte *ptr) {
                                auto object_ptrs =
                                    reinterpret_cast<object_node **>(ptr);
                                nncase_object_release(object_ptrs[0]);
                                delete[] object_ptrs;
                            };

                            auto runtime_tensor =
                                hrt::create(ref_type, {}, bytes_span, deleter)
                                    .unwrap_or_throw();

                            auto host_buffer = runtime_tensor.impl()
                                                   ->buffer()
                                                   .as_host()
                                                   .unwrap_or_throw();
                            auto mapped_data =
                                host_buffer.map(map_write).unwrap_or_throw();
                            auto dest_buffer =
                                as_span<object_node *>(mapped_data.buffer());
                            dest_buffer[0] = object(obj).detach();
                            return runtime_tensor;
                        })
            .def_static(
                "from_numpy",
                [](py::array arr) {
                    arr = py::array::ensure(arr, py::array::c_style);
                    auto src_buffer = arr.request();
                    auto datatype = from_dtype(arr);
                    auto tensor =
                        host_runtime_tensor::create(
                            datatype, to_rt_shape(src_buffer.shape),
                            to_rt_strides(src_buffer.itemsize,
                                          src_buffer.strides),
                            std::span(
                                reinterpret_cast<std::byte *>(src_buffer.ptr),
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
                     if (tensor.is_host()) {
                         auto host = tensor.to_host().unwrap_or_throw();
                         auto src_map =
                             std::move(hrt::map(host, runtime::map_read)
                                           .unwrap_or_throw());
                         auto src_buffer = src_map.buffer();
                         return py::array(
                             to_dtype(tensor.impl()->dtype()),
                             to_py_shape(tensor.impl()->dtype(),
                                         tensor.impl()->shape()),
                             to_py_strides(tensor.impl()->dtype(),
                                           tensor.impl()->strides()),
                             src_buffer.data());
                     } else if (tensor.is_device()) {
                         auto new_tensor =
                             host_runtime_tensor::create(tensor.impl()->dtype(),
                                                         tensor.impl()->shape())
                                 .unwrap_or_throw();
                         tensor.copy_to(new_tensor).unwrap_or_throw();

                         auto host = new_tensor.to_host().unwrap_or_throw();
                         auto src_map =
                             std::move(hrt::map(host, runtime::map_read)
                                           .unwrap_or_throw());
                         auto src_buffer = src_map.buffer();

                         return py::array(
                             to_dtype(tensor.impl()->dtype()),
                             to_py_shape(tensor.impl()->dtype(),
                                         tensor.impl()->shape()),
                             to_py_strides(tensor.impl()->dtype(),
                                           tensor.impl()->strides()),
                             src_buffer.data());
                     } else {
                         throw std::runtime_error("Unknown tensor type!");
                     }
                 })
            .def_property_readonly("dtype",
                                   [](runtime_tensor &tensor) {
                                       if (tensor.empty()) {
                                           return py::dtype("empty");
                                       }
                                       return to_dtype(tensor.impl()->dtype());
                                   })
            .def_property_readonly("shape", [](runtime_tensor &tensor) {
                if (tensor.empty()) {
                    return std::vector<pybind11::ssize_t>();
                }
                auto py_shape =
                    to_py_shape(tensor.impl()->dtype(), tensor.impl()->shape());
                if (tensor.impl()->dtype().is_a<vector_type_t>()) {
                    auto vtype =
                        tensor.impl()->dtype().as<vector_type_t>().unwrap();
                    for (auto lane : vtype->lanes()) {
                        py_shape.push_back(lane);
                    }
                }
                return py_shape;
            });

    return rt_class;
}
} // namespace nncase
