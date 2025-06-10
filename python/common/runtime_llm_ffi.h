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
#include "pytype_utils.h"
#include <nncase/compiler.h>
#include <nncase/llm/attention_config.h>
#include <nncase/llm/paged_attention_config.h>
#include <nncase/llm/paged_attention_enums.h>
#include <nncase/llm/paged_attention_kv_cache.h>
#include <nncase/llm/paged_attention_scheduler.h>
#include <nncase/object.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace nncase::runtime;
namespace py = pybind11;

namespace nncase {
void register_runtime_llm_ffi(py::module &m) {
    py::class_<object_node, object>(m, "Object");

    py::enum_<llm::paged_kvcache_dim_kind>(m, "PagedKVCacheDimKind")
        .value("NumBlocks", llm::paged_kvcache_dim_kind::num_blocks)
        .value("NumLayers", llm::paged_kvcache_dim_kind::num_layers)
        .value("KV", llm::paged_kvcache_dim_kind::kv)
        .value("BlockSize", llm::paged_kvcache_dim_kind::block_size)
        .value("NumKVHeads", llm::paged_kvcache_dim_kind::num_kv_heads)
        .value("HeadDim", llm::paged_kvcache_dim_kind::head_dim);

    py::class_<llm::attention_config_node, object_node, llm::attention_config>(
        m, "AttentionConfig")
        .def(py::init([](size_t num_layers, size_t num_kv_heads,
                         size_t head_dim, py::dtype kv_prim_type) {
            return llm::attention_config(std::in_place, num_layers,
                                         num_kv_heads, head_dim,
                                         nncase::from_dtype(kv_prim_type));
        }))
        .def_property(
            "num_layers",
            py::overload_cast<>(&llm::attention_config_node::num_layers,
                                py::const_),
            py::overload_cast<size_t>(&llm::attention_config_node::num_layers))
        .def_property(
            "num_kv_heads",
            py::overload_cast<>(&llm::attention_config_node::num_kv_heads,
                                py::const_),
            py::overload_cast<size_t>(
                &llm::attention_config_node::num_kv_heads))
        .def_property(
            "head_dim",
            py::overload_cast<>(&llm::attention_config_node::head_dim,
                                py::const_),
            py::overload_cast<size_t>(&llm::attention_config_node::head_dim))
        .def_property(
            "kv_type",
            [](const llm::attention_config &self) {
                return nncase::to_dtype(self->kv_prim_type());
            },
            [](llm::attention_config &self, py::dtype value) {
                self->kv_prim_type(nncase::from_dtype(value));
            });

    py::class_<llm::paged_attention_config_node, llm::attention_config_node,
               llm::paged_attention_config>(m, "PagedAttentionConfig")
        .def(py::init(
                 [](size_t num_layers, size_t num_kv_heads, size_t head_dim,
                    py::dtype kv_prim_type, size_t block_size,
                    const std::array<llm::paged_kvcache_dim_kind, 6>
                        cache_layout = {},
                    const std::vector<llm::paged_kvcache_dim_kind> packed_axes =
                        {},
                    const std::vector<size_t> lanes = {},
                    const std::vector<llm::paged_kvcache_dim_kind>
                        sharding_axes = {},
                    const std::vector<std::vector<size_t>> axis_policies = {}) {
                     std::vector<dims_t> policies;
                     for (auto &&item : axis_policies) {
                         policies.emplace_back(item.begin(), item.end());
                     }
                     return llm::paged_attention_config(
                                std::in_place, num_layers, num_kv_heads,
                                head_dim, nncase::from_dtype(kv_prim_type),
                                block_size, cache_layout, packed_axes,
                                dims_t{lanes.begin(), lanes.end()},
                                sharding_axes, policies)
                         .detach();
                 }),
             py::arg("num_layers"), py::arg("num_kv_heads"),
             py::arg("head_dim"), py::arg("kv_type"), py::arg("block_size"),
             py::arg("cache_layout") =
                 std::array<llm::paged_kvcache_dim_kind, 6>{},
             py::arg("packed_axes") =
                 std::vector<llm::paged_kvcache_dim_kind>{},
             py::arg("lanes") = std::vector<size_t>{},
             py::arg("sharding_axes") =
                 std::vector<llm::paged_kvcache_dim_kind>{},
             py::arg("axis_policies") = std::vector<std::vector<size_t>>{})
        .def_property(
            "block_size",
            py::overload_cast<>(&llm::paged_attention_config_node::block_size,
                                py::const_),
            py::overload_cast<size_t>(
                &llm::paged_attention_config_node::block_size))
        .def_property(
            "cache_layout",
            py::overload_cast<>(&llm::paged_attention_config_node::cache_layout,
                                py::const_),
            py::overload_cast<
                const std::array<llm::paged_kvcache_dim_kind, 6> &>(
                &llm::paged_attention_config_node::cache_layout))
        .def_property_readonly(
            "block_layout",
            py::overload_cast<>(&llm::paged_attention_config_node::block_layout,
                                py::const_))
        .def_property(
            "packed_axes",
            [](const llm::paged_attention_config_node &self) {
                auto axes = self.packed_axes();
                return std::vector<llm::paged_kvcache_dim_kind>(axes.begin(),
                                                                axes.end());
            },
            [](llm::paged_attention_config_node &self,
               const std::vector<llm::paged_kvcache_dim_kind> &packed_axes) {
                self.packed_axes(packed_axes);
            })
        .def_property(
            "lanes",
            [](const llm::paged_attention_config_node &self) {
                auto lanes = self.lanes();
                return std::vector<int>(lanes.begin(), lanes.end());
            },
            [](llm::paged_attention_config_node &self,
               const std::vector<int> &lanes) { self.lanes(lanes); })
        .def_property(
            "sharding_axes",
            [](const llm::paged_attention_config_node &self) {
                auto axes = self.sharding_axes();
                return std::vector<llm::paged_kvcache_dim_kind>(axes.begin(),
                                                                axes.end());
            },
            [](llm::paged_attention_config_node &self,
               const std::vector<llm::paged_kvcache_dim_kind> &sharding_axes) {
                self.sharding_axes(sharding_axes);
            })
        .def_property(
            "axis_policies",
            [](const llm::paged_attention_config_node &self) {
                std::vector<std::vector<size_t>> policies;
                for (auto &&item : self.axis_policies()) {
                    policies.push_back(
                        std::vector<size_t>(item.begin(), item.end()));
                }
                return policies;
            },
            [](llm::paged_attention_config_node &self,
               const std::vector<std::vector<size_t>> &axis_policies) {
                std::vector<dims_t> policies(axis_policies.size());
                for (size_t i = 0; i < axis_policies.size(); i++) {
                    policies[i] = dims_t(axis_policies[i].begin(),
                                         axis_policies[i].end());
                }
                self.axis_policies(policies);
            })
        .def("set_axis_policy", [](llm::paged_attention_config_node &self,
                                   int32_t i,
                                   const std::vector<size_t> &axis_policy) {
            if (i < 0 || i >= self.axis_policies().size() ||
                axis_policy.size() > 8) {
                throw std::out_of_range("Index out of range");
            }
            self.axis_policies(i,
                               dims_t(axis_policy.begin(), axis_policy.end()));
        });

    py::class_<llm::paged_attention_kv_cache_node, object_node,
               llm::paged_attention_kv_cache>(m, "PagedAttentionKVCache")
        .def(py::init([](llm::paged_attention_config config) {
            return llm::paged_attention_kv_cache(std::in_place, config, 0, 0,
                                                 tensor(), tensor(), tensor(),
                                                 tensor(), 0, dims_t{})
                .detach();
        }))
        .def_property_readonly(
            "config",
            [](const llm::paged_attention_kv_cache_node &self) {
                return llm::paged_attention_config(self.config());
            })
        .def_property(
            "num_seqs",
            py::overload_cast<>(&llm::paged_attention_kv_cache_node::num_seqs,
                                py::const_),
            py::overload_cast<int32_t>(
                &llm::paged_attention_kv_cache_node::num_seqs))
        .def_property(
            "num_tokens",
            py::overload_cast<>(&llm::paged_attention_kv_cache_node::num_tokens,
                                py::const_),
            py::overload_cast<int32_t>(
                &llm::paged_attention_kv_cache_node::num_tokens))
        .def_property("conversation_id",
                      py::overload_cast<>(
                          &llm::paged_attention_kv_cache_node::conversation_id,
                          py::const_),
                      py::overload_cast<size_t>(
                          &llm::paged_attention_kv_cache_node::conversation_id))
        .def_property(
            "context_lens",
            [](const llm::paged_attention_kv_cache_node &self) {
                return runtime_tensor(self.context_lens());
            },
            [](llm::paged_attention_kv_cache_node &self,
               const runtime_tensor &context_lens) {
                self.context_lens(context_lens.impl());
            })
        .def_property(
            "seq_lens",
            [](const llm::paged_attention_kv_cache_node &self) {
                return runtime_tensor(self.seq_lens());
            },
            [](llm::paged_attention_kv_cache_node &self,
               const runtime_tensor &seq_lens) {
                self.seq_lens(seq_lens.impl());
            })
        .def_property(
            "block_table",
            [](const llm::paged_attention_kv_cache_node &self) {
                return runtime_tensor(self.block_table());
            },
            [](llm::paged_attention_kv_cache_node &self,
               const runtime_tensor &block_table) {
                self.block_table(block_table.impl());
            })
        .def_property(
            "slot_mapping",
            [](const llm::paged_attention_kv_cache_node &self) {
                return runtime_tensor(self.slot_mapping());
            },
            [](llm::paged_attention_kv_cache_node &self,
               const runtime_tensor &slot_mapping) {
                self.slot_mapping(slot_mapping.impl());
            })
        .def_property(
            "num_blocks",
            py::overload_cast<>(&llm::paged_attention_kv_cache_node::num_blocks,
                                py::const_),
            py::overload_cast<int32_t>(
                &llm::paged_attention_kv_cache_node::num_blocks))
        .def("kv_cache",
             [](llm::paged_attention_kv_cache_node &self,
                const std::vector<size_t> &indices,
                const runtime_tensor &kv_storage) {
                 if (indices.size() != self.kv_shape().size()) {
                     throw std::invalid_argument(
                         "Indices size does not match kv_topo size");
                 }
                 self.kv_cache(dims_t(indices.begin(), indices.end()),
                               kv_storage.impl());
             })
        .def("kv_cache",
             [](const llm::paged_attention_kv_cache_node &self,
                const std::vector<size_t> &indices) {
                 return runtime_tensor(
                     self.kv_cache(dims_t(indices.begin(), indices.end())));
             })
        .def_property(
            "kv_topo",
            [](const llm::paged_attention_kv_cache_node &self) {
                return std::vector<size_t>(self.kv_shape().begin(),
                                           self.kv_shape().end());
            },
            [](llm::paged_attention_kv_cache_node &self,
               const std::vector<size_t> &kv_topo) {
                py::print("When re-setting kv_topo, the kv_cache also need",
                          "to be re-set");
                self.kv_shape(dims_t(kv_topo.begin(), kv_topo.end()));
            });

    py::class_<llm::paged_attention_scheduler_node, object_node,
               llm::paged_attention_scheduler>(m, "PagedAttentionScheduler")
        .def(py::init([](llm::paged_attention_config config, size_t num_blocks,
                         size_t max_model_len,
                         const std::vector<int> &hierarchy) {
            return llm::paged_attention_scheduler(std::in_place, config,
                                                  num_blocks, max_model_len,
                                                  hierarchy)
                .detach();
        }))
        .def("schedule", [](llm::paged_attention_scheduler_node &self,
                            const std::vector<long> &session_ids,
                            const std::vector<long> &query_lens) {
            return self.schedule(session_ids, query_lens).detach();
        });
}
} // namespace nncase