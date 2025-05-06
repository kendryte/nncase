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
#include "runtime_function.h"
#include <nncase/llm/paged_attention_kv_cache.h>
#include <nncase/ntt/arch/cpu/runtime.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

#ifdef WIN32
#include <Windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;
using namespace nncase::ntt::runtime;

cpu_runtime_function::cpu_runtime_function(runtime_module &rt_module)
    : runtime_function(rt_module), block_entry_(nullptr) {}

cpu_runtime_function::~cpu_runtime_function() {}

cpu_runtime_module &cpu_runtime_function::module() const noexcept {
    return static_cast<cpu_runtime_module &>(runtime_function::module());
}

result<void> cpu_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    auto text = module().text().subspan(context.header().entrypoint,
                                        context.header().text_size);
    loader_.load(text);
    block_entry_ = (block_entry_t)loader_.entry();
    return ok();
}

result<value_t> cpu_runtime_function::invoke_core(
    std::span<value_t> parameters,
    [[maybe_unused]] value_t return_value) noexcept {
    std::vector<thread_inout_desc> inouts;
    std::vector<thread_paged_attention_kv_cache_desc *> inout_paged_kvcaches;
    for (auto arg : parameters) {
        try_var(t, arg.as<tensor>());
        try_var(hb, t->buffer().as_host());
        try_var(m, hb.map(map_read_write));

        if (t->dtype().is_a<reference_type_t>()) {
            auto rt = t->dtype().as<reference_type_t>().expect(
                "now only support reference value type!");
            auto vt = rt->elemtype().as<value_type_t>().expect(
                "now only support reference value type!");
            if (vt->uuid() == datatype_t::paged_attention_kv_cache->uuid()) {
                auto refspan =
                    as_span<llm::paged_attention_kv_cache_node *>(m.buffer());
                thread_paged_attention_kv_cache_desc *descs =
                    new thread_paged_attention_kv_cache_desc[refspan.size()];
                for (size_t i = 0; i < refspan.size(); i++) {
                    auto &node = refspan[i];
                    auto &desc = descs[i];
                    {
                        auto cfg = node->config();
                        desc.num_layers = cfg->num_layers();
                        desc.num_kv_heads = cfg->num_kv_heads();
                        desc.head_dim = cfg->head_dim();
                        desc.kv_prim_size = typecode_bytes(cfg->kv_type());

                        // paged config parameters
                        desc.block_size = cfg->block_size();
                        for (size_t i = 0; i < cfg->cache_layout().size();
                             i++) {
                            desc.cache_layout[i] =
                                (int32_t)cfg->cache_layout()[i];
                        }
                        for (size_t i = 0; i < cfg->block_layout().size();
                             i++) {
                            desc.block_layout[i] =
                                (int32_t)cfg->block_layout()[i];
                        }
                        for (size_t i = 0; i < desc.packed_axes.size(); i++) {
                            desc.packed_axes[i] =
                                i < cfg->packed_axes().size()
                                    ? (int32_t)cfg->packed_axes()[i]
                                    : -1;
                        }
                        for (size_t i = 0; i < desc.lanes.size(); i++) {
                            desc.lanes[i] = i < cfg->lanes().size()
                                                ? (int32_t)cfg->lanes()[i]
                                                : -1;
                        }
                        for (size_t i = 0; i < desc.topology.size(); i++) {
                            desc.topology[i] = i < cfg->topology().size()
                                                   ? (int32_t)cfg->topology()[i]
                                                   : -1;
                        }

                        // Basic parameters from attention_kv_cache
                        desc.num_seqs = node->num_seqs();
                        desc.num_tokens = node->num_tokens();
                        {
                            try_var(hbf,
                                    node->context_lens()->buffer().as_host());
                            try_var(mbf, hbf.map(map_read_write));
                            desc.context_lens = (int64_t *)mbf.buffer().data();
                            desc.context_lens_size =
                                mbf.buffer().size_bytes() / sizeof(int64_t);
                        }
                        {
                            try_var(hbf, node->seq_lens()->buffer().as_host());
                            try_var(mbf, hbf.map(map_read_write));
                            desc.seq_lens = (int64_t *)mbf.buffer().data();
                            desc.seq_lens_size =
                                mbf.buffer().size_bytes() / sizeof(int64_t);
                        }

                        // Paged attention specific parameters
                        {
                            try_var(hbf,
                                    node->block_table()->buffer().as_host());
                            try_var(mbf, hbf.map(map_read_write));
                            desc.block_table = (int64_t *)mbf.buffer().data();
                            desc.block_table_shape[0] =
                                node->block_table()->shape()[0];
                            desc.block_table_shape[1] =
                                node->block_table()->shape()[1];
                            desc.block_table_shape[2] =
                                node->block_table()->shape()[2];
                        }
                        {
                            try_var(hbf,
                                    node->slot_mapping()->buffer().as_host());
                            try_var(mbf, hbf.map(map_read_write));
                            desc.slot_mapping = (int64_t *)mbf.buffer().data();
                            desc.slot_mapping_shape[0] =
                                node->slot_mapping()->shape()[0];
                            desc.slot_mapping_shape[1] =
                                node->slot_mapping()->shape()[1];
                        }
                        desc.num_blocks = node->num_blocks();

                        {
                            auto kv_storages = node->kv_storages();
                            for (size_t i = 0; i < kv_storages.size(); i++) {
                                try_var(hbf,
                                        kv_storages[i]->buffer().as_host());
                                try_var(mbf, hbf.map(map_read_write));
                                desc.kv_storages[i] =
                                    (intptr_t)mbf.buffer().data();
                            }
                            for (size_t i = 0; i < desc.kv_shape.size(); i++) {
                                desc.kv_shape[i] =
                                    i < node->kv_shape().size()
                                        ? (int32_t)node->kv_shape()[i]
                                        : -1;
                            }
                        }
                    }
                }
                inout_paged_kvcaches.push_back(descs);
                inouts.emplace_back(thread_inout_desc{
                    .data = (std::byte *)descs,
                    .size = sizeof(intptr_t) * refspan.size(),
                    .shape = t->shape().data(),
                    .strides = t->strides().data(),
                });
            } else {
                return err(std::errc::not_supported);
            }
        } else {
            inouts.emplace_back(thread_inout_desc{
                .data = m.buffer().data(),
                .size = m.buffer().size(),
                .shape = t->shape().data(),
                .strides = t->strides().data(),
            });
        }
        m.release();
    }

    try_(run(inouts));

    for (auto arg : parameters) {
        try_var(t, arg.as<tensor>());
        try_var(hb, t->buffer().as_host());
        try_(hb.unmap());
    }

    return ok(tuple(std::in_place));
}
