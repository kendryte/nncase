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
#include "runtime_llm_ffi.h"

using namespace nncase::runtime;
namespace py = pybind11;

namespace nncase {

inline void register_ref_llm_ffi(py::module &m) {
    py::enum_<llm::attention_dim_kind>(m, "AttentionDimKind")
        .value("Seq", llm::attention_dim_kind::seq)
        .value("Head", llm::attention_dim_kind::head)
        .value("Dim", llm::attention_dim_kind::dim);

    py::class_<clr::ref_paged_attention_kv_cache>(m, "RefPagedAttentionKVCache")
        .def_property_readonly("num_seqs",
                               &clr::ref_paged_attention_kv_cache::num_seqs)
        .def_property_readonly("num_tokens",
                               &clr::ref_paged_attention_kv_cache::num_tokens)
        .def_property_readonly("num_blocks",
                               &clr::ref_paged_attention_kv_cache::num_blocks)
        .def_property_readonly("context_lens",
                               &clr::ref_paged_attention_kv_cache::context_lens)
        .def_property_readonly("seq_lens",
                               &clr::ref_paged_attention_kv_cache::seq_lens)
        .def_property_readonly("block_table",
                               &clr::ref_paged_attention_kv_cache::block_table)
        .def_property_readonly("slot_mapping",
                               &clr::ref_paged_attention_kv_cache::slot_mapping)
        .def_property_readonly("kv_caches",
                               &clr::ref_paged_attention_kv_cache::kv_caches)
        .def("as_ivalue", &clr::ref_paged_attention_kv_cache::as_ivalue)
        .def("as_rtvalue", &clr::ref_paged_attention_kv_cache::as_rtvalue)
        .def("dump_json", &clr::ref_paged_attention_kv_cache::dump_json);

    py::class_<clr::ref_paged_attention_scheduler>(m,
                                                   "RefPagedAttentionScheduler")
        .def(py::init([](llm::paged_attention_config config, int32_t num_blocks,
                         int32_t max_model_len,
                         const std::vector<int32_t> &hierarchy) {
            return clr::ref_paged_attention_scheduler(config, num_blocks,
                                                      max_model_len, hierarchy);
        }))
        .def("schedule",
             [](clr::ref_paged_attention_scheduler &self,
                const std::vector<int64_t> &session_ids,
                const std::vector<int64_t> &query_lens) {
                 if (session_ids.size() != query_lens.size()) {
                     throw std::runtime_error(
                         "session_ids and query_lens must have the same "
                         "length");
                 }
                 return self.schedule(session_ids, query_lens);
             })
        .def("create_test_function",
             [](clr::ref_paged_attention_scheduler &self, int32_t num_q_heads,
                const std::vector<llm::attention_dim_kind> &q_layout,
                const std::vector<llm::attention_dim_kind> &kv_layout) {
                 if (q_layout.size() != kv_layout.size() ||
                     q_layout.size() != 3) {
                     throw std::runtime_error(
                         "layout size must be 3 and equal");
                 }
                 return self.create_test_function(num_q_heads, q_layout,
                                                  kv_layout);
             });
}

} // namespace nncase