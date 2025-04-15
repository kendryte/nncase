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
// #include "nncase/runtime/duca_paged_attention_kv_cache.h"
// #include <nncase/runtime/interpreter_for_causal_lm.h>
#include "nncase/attention_config.h"
#include "type_casters.h"
#include <nncase/object.h>
#include <nncase/paged_attention_config.h>
#include <nncase/paged_attention_kv_cache.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

using namespace nncase::runtime;
namespace py = pybind11;

namespace nncase {
inline void register_llm(py::module &m) {
    py::class_<object_node, object>(m, "Object");

    py::class_<attention_config_node, object_node, attention_config>(
        m, "AttentionConfig")
        .def(py::init([](size_t num_layers, size_t num_kv_heads,
                         size_t head_dim, nncase::typecode_t kv_type) {
            return attention_config(std::in_place, num_layers, num_kv_heads,
                                    head_dim, kv_type);
        }))
        .def_property(
            "num_layers",
            py::overload_cast<>(&attention_config_node::num_layers, py::const_),
            py::overload_cast<size_t>(&attention_config_node::num_layers))
        .def_property(
            "num_kv_heads",
            py::overload_cast<>(&attention_config_node::num_kv_heads,
                                py::const_),
            py::overload_cast<size_t>(&attention_config_node::num_kv_heads))
        .def_property(
            "head_dim",
            py::overload_cast<>(&attention_config_node::head_dim, py::const_),
            py::overload_cast<size_t>(&attention_config_node::head_dim));

    py::class_<paged_attention_config_node, attention_config_node,
               paged_attention_config>(m, "PagedAttentionConfig")
        .def(
            py::init([](size_t num_layers, size_t num_kv_heads, size_t head_dim,
                        nncase::typecode_t kv_type, size_t block_size) {
                return paged_attention_config(std::in_place, num_layers,
                                              num_kv_heads, head_dim, kv_type,
                                              block_size);
            }))
        .def_property("block_size",
                      py::overload_cast<>(
                          &paged_attention_config_node::block_size, py::const_),
                      py::overload_cast<size_t>(
                          &paged_attention_config_node::block_size));

    // py::class_<paged_attention_scheduler>(m, "PagedAttentionScheduler")
    //     .def(py::init<int>())
    //     .def("initialize", &paged_attention_scheduler::initialize)
    //     .def("schedule", &paged_attention_scheduler::schedule);
}
} // namespace nncase

// PYBIND11_MAKE_OPAQUE(std::vector<runtime_tensor>)
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<runtime_tensor>>)
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<runtime_tensor>>>)

// inline void register_llm_interpreter(py::module &m) {
//     py::class_<interpreter_for_causal_lm>(m, "InterpreterForCausalLM")
//         .def(py::init<>())
//         .def("load_model",
//              [](interpreter &interp, std::span<const std::byte> buffer) {
//                  interp.load_model(buffer, false).unwrap_or_throw();
//              })
//         .def("load_model",
//              [](interpreter &interp, nncase::runtime::stream &stream) {
//                  interp.load_model(stream).unwrap_or_throw();
//              })
//         .def("__call__", [](py::object input_ids_obj, py::object
//         input_mask_obj,
//                             py::object position_ids_obj,
//                             py::object attention_kv_cache_obj,
//                             NNCASE_UNUSED py::kwargs kwargs) {
//             runtime_tensor input_ids;

//             if (py::isinstance<runtime_tensor>(input_ids_obj)) {
//                 input_ids = py::cast<runtime_tensor>(input_ids_obj);
//             } else {
//                 throw py::type_error(
//                     "input_ids must be a runtime_tensor or None");
//             }

//             std::optional<runtime_tensor> input_mask;
//             if (py::isinstance<py::none>(input_mask_obj)) {
//             } else if (py::isinstance<runtime_tensor>(input_mask_obj)) {
//                 input_mask = py::cast<runtime_tensor>(input_mask_obj);
//             }

//             std::optional<runtime_tensor> position_ids;
//             if (py::isinstance<py::none>(position_ids_obj)) {
//             } else if (py::isinstance<runtime_tensor>(position_ids_obj)) {
//                 position_ids = py::cast<runtime_tensor>(position_ids_obj);
//             }

//             std::optional<attention_kv_cache> kv_cache;
//             if (py::isinstance<py::none>(attention_kv_cache_obj)) {
//             } else if (py::isinstance<attention_kv_cache>(
//                            attention_kv_cache_obj)) {
//                 kv_cache =
//                 py::cast<attention_kv_cache>(attention_kv_cache_obj);
//             }

//             return py::none();
//         });
// }

// inline void register_kv_cache(py::module &m) {

//     // todo bind vector
//     // py::bind_vector<std::vector<runtime_tensor>>(m, "RuntimeTensorList");
//     // py::bind_vector<std::vector<std::vector<runtime_tensor>>>(
//     //     m, "RuntimeTensorMatrix");
//     //
//     py::bind_vector<std::vector<std::vector<std::vector<runtime_tensor>>>>(
//     //     m, "RuntimeTensorCube");
//     // py::class_<attention_kv_cache>(m, "AttentionKVCache");

//     py::class_<duca_paged_attention_kv_cache, attention_kv_cache>(
//         m, "DUCAPagedAttentionKVCache")
//         .def(py::init())
//         .def_readwrite("num_prefills",
//         &duca_paged_attention_kv_cache::num_prefills)
//         .def_readwrite("num_prefill_tokens",
//                        &duca_paged_attention_kv_cache::num_prefill_tokens)
//         .def_readwrite("num_decode_tokens",
//                        &duca_paged_attention_kv_cache::num_decode_tokens)
//         .def_readwrite("block_tables",
//         &duca_paged_attention_kv_cache::block_tables)
//         .def_readwrite("slot_mapping",
//         &duca_paged_attention_kv_cache::slot_mapping) .def_property(
//             "kv_caches",
//             [](const duca_paged_attention_kv_cache &self) {
//                 auto value = py::list();
//                 if (self.kv_caches.empty()) {
//                     return value;
//                 }

//                 for (size_t i = 0; i < self.kv_caches.size(); i++) {
//                     auto value_i = py::list();
//                     for (size_t j = 0; j < self.kv_caches[i].size(); j++) {
//                         auto value_j = py::list();
//                         for (size_t k = 0; k < self.kv_caches[i][j].size();
//                              k++) {
//                             value_j.append(self.kv_caches[i][j][k]);
//                         }
//                         value_i.append(value_j);
//                     }
//                     value.append(value_i);
//                 }
//                 return value;
//             },
//             [](duca_paged_attention_kv_cache &self, const py::list &value) {
//                 if (!self.kv_caches.empty()) {
//                     throw py::value_error(
//                         "can't assgin kv caches when it is not empty.");
//                 }

//                 self.kv_caches.resize(value.size());
//                 for (size_t i = 0; i < value.size(); i++) {
//                     auto value_i = py::cast<py::list>(value[i]);
//                     self.kv_caches[i].resize(value_i.size());
//                     for (size_t j = 0; j < value.size(); j++) {
//                         auto value_j = py::cast<py::list>(value_i[j]);
//                         self.kv_caches[i][j].resize(value_j.size());
//                         for (size_t k = 0; k < value.size(); k++) {
//                             self.kv_caches[i][j][k] =
//                                 py::cast<runtime_tensor>(value_j[k]);
//                         }
//                     }
//                 }
//             });
// }