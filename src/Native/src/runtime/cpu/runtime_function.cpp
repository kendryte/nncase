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
#include "nncase/runtime/buffer.h"
#include "nncase/tensor.h"
#include "nncase/value.h"
#include <cstdint>
#include <nncase/llm/paged_attention_kv_cache.h>
#include <nncase/ntt/arch/cpu/runtime.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <nncase/type.h>
#include <utility>
#include <vector>

#ifdef WIN32
#include <Windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;
using namespace nncase::ntt::runtime;

typedef struct {
    uint32_t output_align;
    uint32_t local_data_align;
    uint64_t output_pool_size;
    uint64_t local_data_pool_size;
} kernel_desc_header;

cpu_runtime_function::cpu_runtime_function(runtime_module &rt_module)
    : runtime_function(rt_module), block_entry_(nullptr) {}

cpu_runtime_function::~cpu_runtime_function() {}

cpu_runtime_module &cpu_runtime_function::module() const noexcept {
    return static_cast<cpu_runtime_module &>(runtime_function::module());
}

result<void> cpu_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    try_(context.read_section(
        ".desc", [this](auto reader, size_t) -> result<void> {
            auto header = reader.template read<kernel_desc_header>();

            // Allocate output buffer
            buffer_allocate_options options{};
            options.flags = HOST_BUFFER_ALLOCATE_CPU_ONLY;
            options.alignment = header.output_align;
            try_var(output_buffer, buffer_allocator::host().allocate(
                                       header.output_pool_size, options));
            try_set(this->output_buffer_,
                    output_buffer.template as<host_buffer_t>());

            // Allocate local datas
            options.alignment = header.local_data_align;
            auto threads_count =
                module().cdim() * module().bdim() * module().tdim();
            local_datas_.resize(threads_count);
            for (size_t i = 0; i < threads_count; i++) {
                try_var(buffer, buffer_allocator::host().allocate(
                                    header.local_data_pool_size, options));
                try_set(local_datas_[i],
                        output_buffer.template as<host_buffer_t>());
            }
            return ok();
        }));
    auto text = module().text().subspan(context.header().entrypoint,
                                        context.header().text_size);
    loader_.load(text);
    block_entry_ = (block_entry_t)loader_.entry();

    // Allocate input descs
    auto input_size = parameters_size();
    input_descs_.resize(input_size);

    // Allocate output descs
    auto output_size = return_size();
    output_descs_.resize(output_size);
    output_shapes_.resize(output_size);
    output_strides_.resize(output_size);
    for (size_t i = 0; i < output_size; i++) {
        try_var(type, return_type(i));
        try_var(ttype, type.as<tensor_type>());
        auto rank = ttype->shape().rank();
        CHECK_WITH_ERR(rank.has_value(), std::errc::invalid_argument);
        output_shapes_[i].resize(*rank);
        output_strides_[i].resize(*rank);
        output_descs_[i] = thread_inout_desc{
            .data = nullptr,
            .size = 0,
            .shape = output_shapes_[i].data(),
            .strides = output_strides_[i].data(),
        };
    }

    return ok();
}

result<value_t> cpu_runtime_function::invoke_core(
    std::span<value_t> parameters,
    [[maybe_unused]] value_t return_value) noexcept {
    size_t input_id = 0;
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
                input_descs_[input_id++] = thread_inout_desc{
                    .data = (std::byte *)descs,
                    .size = sizeof(thread_paged_attention_kv_cache_desc) *
                            refspan.size(),
                    .shape = const_cast<size_t *>(t->shape().data()),
                    .strides = const_cast<size_t *>(t->strides().data()),
                };
            } else {
                return err(std::errc::not_supported);
            }
        } else {
            input_descs_[input_id++] = thread_inout_desc{
                .data = m.buffer().data(),
                .size = m.buffer().size(),
                .shape = const_cast<size_t *>(t->shape().data()),
                .strides = const_cast<size_t *>(t->strides().data()),
            };
        }
        m.release();
    }

    try_var(mapped_output, output_buffer_->map(map_read_write));
    auto output_data = mapped_output.buffer().data();

    try_(run(output_data));

    std::vector<value_t> outputs(return_size());
    for (size_t i = 0; i < outputs.size(); i++) {
        try_set(outputs[i], create_output_tensor(i, parameters, output_data));
    }

    for (auto arg : parameters) {
        try_var(t, arg.as<tensor>());
        try_var(hb, t->buffer().as_host());
        try_(hb.unmap());
    }

    for (auto ptrs : inout_paged_kvcaches) {
        delete[] ptrs;
    }

    auto output_value = outputs.size() == 1
                            ? outputs[0]
                            : tuple(std::in_place, std::move(outputs));
    return ok(output_value);
}

result<tensor>
cpu_runtime_function::create_output_tensor(size_t output_id,
                                           std::span<value_t> parameters,
                                           std::byte *output_data) noexcept {
    auto &output_desc = output_descs_[output_id];
    buffer_slice buffer;
    intptr_t offset;
    // 1. Find in inputs
    for (size_t i = 0; i < input_descs_.size(); i++) {
        auto &candidate_desc = input_descs_[i];
        if (candidate_desc.data <= output_desc.data &&
            candidate_desc.data + candidate_desc.size >=
                output_desc.data + output_desc.size) {
            try_var(t, parameters[i].as<tensor>());
            buffer = t->buffer();
            offset = output_desc.data - candidate_desc.data;
            break;
        }
    }

    // 2. Find in output buffer
    if (buffer.buffer().empty()) {
        if (output_data <= output_desc.data &&
            output_data + output_buffer_->size_bytes() >=
                output_desc.data + output_desc.size) {
            buffer = buffer_slice(output_buffer_);
            offset = output_desc.data - output_data;
        }
    }

    if (buffer.buffer().empty()) {
        return err(std::errc::invalid_argument);
    }

    // 2. Fix offset & size
    buffer = buffer_slice(buffer.buffer(), buffer.start() + offset,
                          output_desc.size);
    try_var(output_type, return_type(output_id));
    try_var(ttype, output_type.as<tensor_type>());
    return ok(tensor(std::in_place, ttype->dtype(), output_shapes_[output_id],
                     output_strides_[output_id], buffer));
}
