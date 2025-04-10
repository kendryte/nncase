#pragma once
// #include "nncase/runtime/duca_paged_attention_kv_cache.h"
// #include <nncase/runtime/interpreter_for_causal_lm.h>
#include <nncase/attention_kv_cache.h>
#include <nncase/paged_attention_config.h>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

using namespace nncase::runtime;
namespace py = pybind11;

class session_info {
  public:
    int64_t slot_start;
    int64_t slot_end;
    int64_t context_len;
};

class paged_attention_scheduler {
  public:
    paged_attention_scheduler(int max_model_len)
        : max_model_len_(max_model_len), session_infos_() {}

    void initialize(nncase::paged_attention_config config, int num_blocks) {
        config_ = config;
        num_blocks_ = num_blocks;
        kv_cache_ =
            hrt::create(nncase::dt_float32,
                        {2, (size_t)config_->num_layers, (size_t)num_blocks_,
                         (size_t)config_->num_kv_heads,
                         (size_t)config_->head_dim},
                        host_runtime_tensor::memory_pool_t::pool_shared)
                .unwrap_or_throw();
    }

    nncase::attention_kv_cache schedule(runtime_tensor session_ids,
                                        runtime_tensor tokens_count) {

        if (session_ids.shape()[0] > 1) {
            throw std::runtime_error("not support multi user");
        }
        auto session_ids_buffer =
            session_ids.impl()->buffer().as_host().unwrap_or_throw();
        auto mapped_session_ids_buffer =
            session_ids_buffer.map(map_access_t::map_read).unwrap_or_throw();
        auto session_ids_ptr = reinterpret_cast<int64_t *>(
            mapped_session_ids_buffer.buffer().data());

        auto tokens_count_buffer =
            tokens_count.impl()->buffer().as_host().unwrap_or_throw();
        auto mapped_tokens_count_buffer =
            tokens_count_buffer.map(map_access_t::map_read).unwrap_or_throw();
        auto tokens_count_ptr = reinterpret_cast<int64_t *>(
            mapped_tokens_count_buffer.buffer().data());

        // hrt::create(nncase::dt_int64, session_ids.shape(), , true);
        std::vector<int64_t> seq_lens(session_ids.shape()[0]);
        std::vector<int64_t> context_lens(session_ids.shape()[0]);
        std::vector<int64_t> slot_maping(std::accumulate(
            tokens_count_ptr, tokens_count_ptr + session_ids.shape()[0], 0));
        int64_t max_seq_len = 0;
        size_t query_token_index = 0;
        for (size_t i = 0; session_ids.shape()[0]; i++) {
            auto session_id = session_ids_ptr[i];
            auto token_count = tokens_count_ptr[i];
            if (session_infos_.find(session_id) == session_infos_.end()) {
                session_infos_[session_id] = session_info{
                    session_id * max_model_len_,
                    (session_id + 1) * max_model_len_,
                    0,
                };
            }
            auto &info = session_infos_[session_id];
            context_lens[i] = info.context_len;
            seq_lens[i] = info.context_len + token_count;
            if (seq_lens[i] > max_model_len_) {
                throw std::runtime_error(
                    "the seq lens is large than max model length!");
            }
            max_seq_len = std::max(max_seq_len, seq_lens[i]);
            for (size_t j = query_token_index; j < token_count; j++) {
                slot_maping[j] = info.slot_start + info.context_len +
                                 (j - query_token_index);
            }
        }
        auto max_block_nums =
            (max_seq_len + (config_->block_size - 1)) / config_->block_size;
        std::vector<int64_t> block_tables(session_ids.shape()[0] *
                                          max_block_nums);
        for (size_t i = 0; session_ids.shape()[0]; i++) {
            auto session_id = session_ids_ptr[i];
            auto &info = session_infos_[session_id];
            for (size_t j = 0; j < (seq_lens[i] + (config_->block_size - 1)) /
                                       config_->block_size;
                 j++) {
                block_tables[i * max_block_nums + j] =
                    (info.slot_start / config_->block_size) + j;
            }
        }

        // convert vector to tensor
        // clang-format off
        auto seq_lens_tensor = hrt::create(nncase::dt_int64, session_ids.shape(), std::span<std::byte>(reinterpret_cast<std::byte *>(seq_lens.data()), seq_lens.size() * sizeof(int64_t)), true).unwrap_or_throw(); 
        auto context_lens_tensor = hrt::create(nncase::dt_int64, session_ids.shape(), std::span<std::byte>( reinterpret_cast<std::byte *>(context_lens.data()), context_lens.size() * sizeof(int64_t)), true).unwrap_or_throw(); 
        auto slot_maping_tensor = hrt::create(nncase::dt_int64, session_ids.shape(), std::span<std::byte>( reinterpret_cast<std::byte *>(slot_maping.data()), slot_maping.size() * sizeof(int64_t)), true).unwrap_or_throw(); 
        auto block_tables_tensor = hrt::create(nncase::dt_int64, {session_ids.shape()[0], (size_t)max_block_nums}, std::span<std::byte>( reinterpret_cast<std::byte *>(block_tables.data()), block_tables.size() * sizeof(int64_t)), true).unwrap_or_throw();
        // clang-format on

        return nncase::attention_kv_cache(
            std::in_place, kv_cache_, seq_lens_tensor, context_lens_tensor,
            block_tables_tensor, slot_maping_tensor);
    }

  private:
    nncase::paged_attention_config config_;
    int num_blocks_;
    runtime_tensor kv_cache_;
    int64_t max_model_len_;
    std::unordered_map<int64_t, session_info> session_infos_;
};

inline void register_paged_attention_scheduler(py::module &m) {
    py::class_<nncase::attention_config>(m, "AttentionConfig")
        .def("__init__",
             [](int num_layers, int num_kv_heads, int head_dim) {
                 return nncase::attention_config(std::in_place, num_layers,
                                                 num_kv_heads, head_dim);
             })
        .def_property(
            "num_layers",
            [](const nncase::attention_config &cfg) { return cfg->num_layers; },
            [](nncase::attention_config &cfg, int num_layers) {
                cfg->num_layers = num_layers;
            })
        .def_property(
            "num_kv_heads",
            [](const nncase::attention_config &cfg) {
                return cfg->num_kv_heads;
            },
            [](nncase::attention_config &cfg, int num_kv_heads) {
                cfg->num_kv_heads = num_kv_heads;
            })
        .def_property(
            "head_dim",
            [](const nncase::attention_config &cfg) { return cfg->head_dim; },
            [](nncase::attention_config &cfg, int head_dim) {
                cfg->head_dim = head_dim;
            });

    py::class_<nncase::paged_attention_config>(m, "PagedAttentionConfig")
        .def(
            "__init__",
            [](int num_layers, int num_kv_heads, int head_dim, int block_size) {
                return nncase::paged_attention_config(std::in_place, num_layers,
                                                      num_kv_heads, head_dim,
                                                      block_size);
            })
        .def_property(
            "block_size",
            [](const nncase::paged_attention_config &cfg) {
                return cfg->block_size;
            },
            [](nncase::paged_attention_config &cfg, int block_size) {
                cfg->block_size = block_size;
            });

    py::class_<paged_attention_scheduler>(m, "PagedAttentionScheduler")
        .def(py::init<int>())
        .def("initialize", &paged_attention_scheduler::initialize)
        .def("schedule", &paged_attention_scheduler::schedule);
}

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