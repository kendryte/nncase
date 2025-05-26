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
#include <nncase/api.h>
#include <nncase/llm/paged_attention_config.h>
#include <nncase/llm/paged_attention_kv_cache.h>
#include <nncase/llm/paged_attention_scheduler.h>
#include <nncase/object.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#ifndef _WIN32
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
#endif

using namespace nncase;
using namespace nncase::runtime;

namespace {
#ifndef _WIN32
volatile bool g_wait_for_debugger = false;
#endif

#define c_try(x)                                                               \
    {                                                                          \
        auto v = (x);                                                          \
        if (!v.is_ok())                                                        \
            return -v.unwrap_err().value();                                    \
    }

#define c_try_var(name, x)                                                     \
    typename decltype((x))::value_type name;                                   \
    {                                                                          \
        auto v = (x);                                                          \
        if (v.is_ok())                                                         \
            name = std::move(v.unwrap());                                      \
        else                                                                   \
            return -v.unwrap_err().value();                                    \
    }

#define c_try_set(name, x)                                                     \
    {                                                                          \
        auto v = (x);                                                          \
        if (v.is_ok())                                                         \
            name = std::move(v.unwrap());                                      \
        else                                                                   \
            return -v.unwrap_err().value();                                    \
    }

result<dims_t> to_dims(const uint32_t *dims, uint32_t length) {
    CHECK_WITH_ERR(dims || !length, std::errc::invalid_argument);
    dims_t d(length);
    for (size_t i = 0; i < length; i++) {
        d[i] = (size_t)dims[i];
    }
    return ok(std::move(d));
}

result<strides_t> to_strides(const uint32_t *strides, uint32_t length) {
    CHECK_WITH_ERR(strides || !length, std::errc::invalid_argument);
    strides_t s(length);
    for (size_t i = 0; i < length; i++) {
        s[i] = (size_t)strides[i];
    }
    return ok(std::move(s));
}
} // namespace

extern "C" {
int nncase_object_add_ref(nncase::object_node *node) {
    if (node)
        node->add_ref();
    return 0;
}

int nncase_object_release(nncase::object_node *node) {
    if (node)
        return node->release();
    return 0;
}

int nncase_interp_create(nncase::runtime::interpreter **interp) {
    if (interp) {
        *interp = new interpreter();
        return 0;
    }
    return -EINVAL;
}

int nncase_interp_free(nncase::runtime::interpreter *interp) {
    if (interp) {
        delete interp;
        return 0;
    }
    return -EINVAL;
}

int nncase_interp_load_model(nncase::runtime::interpreter *interp,
                             void *model_buffer, uint32_t model_size,
                             bool copy_buffer) {
    if (interp) {
        c_try(interp->load_model(
            {reinterpret_cast<const std::byte *>(model_buffer), model_size},
            copy_buffer));
        return 0;
    }
    return -EINVAL;
}

int nncase_interp_load_model_from_path(nncase::runtime::interpreter *interp,
                                       const char *model_path) {
    if (interp) {
        std::ifstream ifs(model_path, std::ios::in | std::ios::binary);
        nncase::runtime::std_istream stream(ifs);
        c_try(interp->load_model(stream));
        ifs.close();
        return 0;
    }
    return -EINVAL;
}

int nncase_interp_set_dump_root(nncase::runtime::interpreter *interp,
                                const char *path) {
    if (interp && path) {
        interp->options().set("dump_root", path);
#ifndef NNCASE_BAREMETAL
        interp->dump_manager()->set_dump_root(path);
#endif
        // todo:set dump level
        return 0;
    }
    return -EINVAL;
}

int nncase_interp_get_entry_func(nncase::runtime::interpreter *interp,
                                 nncase::runtime::runtime_function **func) {
    if (interp && func) {
        c_try_var(entry, interp->entry_function());
        *func = entry;
        return 0;
    }
    return -EINVAL;
}

int nncase_func_get_params_size(nncase::runtime::runtime_function *func,
                                uint32_t *size) {
    if (func && size) {
        *size = func->parameters_size();
        return 0;
    }
    return -EINVAL;
}

int nncase_func_invoke(nncase::runtime::runtime_function *func,
                       value_node **params, uint32_t params_size,
                       value_node **result) {
    if (func && (params || !params_size) && result) {
        std::span<value_t> param_values{reinterpret_cast<value_t *>(params),
                                        params_size};
        c_try_var(retval, func->invoke(param_values));
        *result = retval.detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_buffer_allocator_get_host(
    nncase::runtime::buffer_allocator **alloc) {
    if (alloc) {
        *alloc = &buffer_allocator::host();
        return 0;
    }
    return -EINVAL;
}

int nncase_buffer_allocator_alloc(nncase::runtime::buffer_allocator *alloc,
                                  uint32_t bytes,
                                  [[maybe_unused]] void *options,
                                  nncase::runtime::buffer_node **buffer) {
    if (alloc && buffer) {
        c_try_var(buf, alloc->allocate(bytes, {}));
        *buffer = buf.detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_buffer_as_host(nncase::runtime::buffer_node *buffer,
                          nncase::runtime::host_buffer_node **host_buffer) {
    if (buffer && host_buffer) {
        c_try_var(hbuf, buffer_t(buffer).as<host_buffer_t>());
        *host_buffer = hbuf.detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_host_buffer_map(nncase::runtime::host_buffer_node *host_buffer,
                           nncase::runtime::map_access_t access, void **data,
                           uint32_t *bytes) {
    if (host_buffer) {
        c_try_var(mapped_b, host_buffer->map(access));
        if (data)
            *data = mapped_b.buffer().data();
        if (bytes)
            *bytes = (uint32_t)mapped_b.buffer().size_bytes();
        mapped_b.release();
        return 0;
    }
    return -EINVAL;
}

int nncase_host_buffer_unmap(nncase::runtime::host_buffer_node *host_buffer) {
    if (host_buffer) {
        c_try(host_buffer->unmap());
        return 0;
    }
    return -EINVAL;
}

int nncase_dtype_create_prime(nncase::typecode_t typecode,
                              nncase::prim_type_node **dtype) {
    if (dtype) {
        c_try_var(type, datatype_t::from_typecode(typecode));
        *dtype = type.detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_dtype_create_vector(nncase::datatype_node *elem_type, int32_t *lanes,
                               int32_t length,
                               nncase::vector_type_node **dtype) {
    if (dtype) {
        *dtype = vector_type_t(std::in_place, elem_type,
                               dims_t{lanes, lanes + length})
                     .detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_dtype_get_typecode(nncase::datatype_node *dtype) {
    return dtype->typecode();
}

int nncase_vector_dtype_get_elem_type(nncase::vector_type_node *handle,
                                      nncase::datatype_node **elemType) {
    if (handle && elemType) {
        *elemType = datatype_t(handle->elemtype()).detach();
        return 0;
    }

    return -EINVAL;
}

int nncase_vector_dtype_get_lanes_length(nncase::vector_type_node *handle,
                                         int32_t *length) {
    if (handle && length) {
        *length = handle->lanes().size();
        return 0;
    }
    return -EINVAL;
}

int nncase_vector_dtype_get_lanes(nncase::vector_type_node *handle,
                                  int32_t *lanes) {
    if (handle && lanes) {
        auto lanes_span = handle->lanes();
        for (size_t i = 0; i < lanes_span.size(); i++) {
            lanes[i] = lanes_span[i];
        }
        return 0;
    }
    return -EINVAL;
}

int nncase_dtype_create_reference(nncase::datatype_node *elem_type,
                                  nncase::reference_type_node **dtype) {
    if (dtype) {
        *dtype = reference_type_t(std::in_place, elem_type).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_reference_dtype_get_elem_type(nncase::reference_type_node *handle,
                                         nncase::datatype_node **elemType) {
    if (handle && elemType) {
        *elemType = datatype_t(handle->elemtype()).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_dtype_create_attention_kv_cache(nncase::datatype_node **dtype) {
    if (dtype) {
        *dtype = datatype_t(datatype_t::attention_kv_cache).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_dtype_create_paged_attention_kv_cache(
    nncase::datatype_node **dtype) {
    if (dtype) {
        *dtype = datatype_t(datatype_t::paged_attention_kv_cache).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_value_dtype_get_uuid(nncase::value_type_node *value_type,
                                uint8_t *uuid, int32_t uuid_length) {
    if (value_type && uuid && uuid_length) {
        auto uuid_span = value_type->uuid();
        if (uuid_span.size() > uuid_length)
            return -EOVERFLOW;
        std::copy(uuid_span.begin(), uuid_span.end(), uuid);
        return 0;
    }
    return -EINVAL;
}

int nncase_value_is_tensor(nncase::value_node *value, bool *is_tensor) {
    if (value && is_tensor) {
        *is_tensor = value_t(value).is_a<tensor>();
        return 0;
    }
    return -EINVAL;
}

int nncase_tensor_create(nncase::datatype_node *dtype, const uint32_t *dims,
                         uint32_t dims_length, const uint32_t *strides,
                         uint32_t strides_length, nncase_buffer_slice *buffer,
                         nncase::tensor_node **tensor) {
    if (dtype && buffer && tensor) {
        c_try_var(d, to_dims(dims, dims_length));
        c_try_var(s, to_strides(strides, strides_length));
        *tensor =
            nncase::tensor(
                std::in_place, dtype, std::move(d), std::move(s),
                buffer_slice(buffer->buffer, buffer->start, buffer->size_bytes))
                .detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_tensor_get_dtype(nncase::tensor_node *tensor,
                            nncase::datatype_node **dtype) {
    if (tensor && dtype) {
        *dtype = datatype_t(tensor->dtype()).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_tensor_get_buffer(nncase::tensor_node *tensor,
                             nncase_buffer_slice *buffer) {
    if (tensor && buffer) {
        auto &slice = tensor->buffer();
        buffer->buffer = buffer_t(slice.buffer()).detach();
        buffer->start = (uint32_t)slice.start();
        buffer->size_bytes = (uint32_t)slice.size_bytes();
        return 0;
    }
    return -EINVAL;
}

int nncase_tensor_get_dims(nncase::tensor_node *tensor, uint32_t *dims,
                           uint32_t *dims_length) {
    if (tensor && dims_length) {
        auto shape = tensor->shape();
        auto required_length = (uint32_t)shape.size();
        if (*dims_length < required_length) {
            *dims_length = required_length;
            if (dims)
                return -EOVERFLOW;
            return 0;
        }

        *dims_length = required_length;
        if (dims) {
            for (size_t i = 0; i < shape.size(); i++) {
                dims[i] = (uint32_t)shape[i];
            }
        }
        return 0;
    }
    return -EINVAL;
}

int nncase_tensor_get_strides(nncase::tensor_node *tensor, uint32_t *strides,
                              uint32_t *strides_length) {
    if (tensor && strides_length) {
        auto src_strides = tensor->strides();
        auto required_length = (uint32_t)src_strides.size();
        if (*strides_length < required_length) {
            *strides_length = required_length;
            return -EOVERFLOW;
        }

        *strides_length = required_length;
        if (strides) {
            for (size_t i = 0; i < src_strides.size(); i++) {
                strides[i] = (uint32_t)src_strides[i];
            }
        }
        return 0;
    }
    return -EINVAL;
}

int nncase_tuple_create(nncase::value_node **fields, uint32_t fields_length,
                        nncase::tuple_node **tuple) {
    if (fields && fields_length && tuple) {
        std::vector<value_t> values(fields_length, nullptr);
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = fields[i];
        }
        *tuple = nncase::tuple(std::in_place, std::move(values)).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_tuple_get_fields(nncase::tuple_node *tuple,
                            nncase::value_node **fields,
                            uint32_t *fields_length) {
    if (tuple && fields_length) {
        auto src_fields = tuple->fields();
        auto required_length = (uint32_t)src_fields.size();
        if (*fields_length < required_length) {
            *fields_length = required_length;
            return -EOVERFLOW;
        }

        *fields_length = required_length;
        if (fields) {
            for (size_t i = 0; i < src_fields.size(); i++) {
                fields[i] = value_t(src_fields[i]).detach();
            }
        }
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_config_create(
    int32_t num_layers, int32_t num_kv_heads, int32_t head_dim,
    nncase::typecode_t kv_type, nncase::llm::attention_config_node **config) {
    if (config) {
        *config = nncase::llm::attention_config(std::in_place, num_layers,
                                                num_kv_heads, head_dim, kv_type)
                      .detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_config_get_num_layers(
    nncase::llm::attention_config_node *config, int32_t *num_layers) {
    if (config && num_layers) {
        *num_layers = config->num_layers();
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_config_set_num_layers(
    nncase::llm::attention_config_node *config, int32_t num_layers) {
    if (config) {
        config->num_layers(num_layers);
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_config_get_num_kv_heads(
    nncase::llm::attention_config_node *config, int32_t *num_kv_heads) {
    if (config && num_kv_heads) {
        *num_kv_heads = config->num_kv_heads();
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_config_set_num_kv_heads(
    nncase::llm::attention_config_node *config, int32_t num_kv_heads) {
    if (config) {
        config->num_kv_heads(num_kv_heads);
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_config_get_head_dim(
    nncase::llm::attention_config_node *config, int32_t *head_dim) {
    if (config && head_dim) {
        *head_dim = config->head_dim();
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_config_set_head_dim(
    nncase::llm::attention_config_node *config, int32_t head_dim) {
    if (config) {
        config->head_dim(head_dim);
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_config_get_kv_type(
    nncase::llm::attention_config_node *config, nncase::typecode_t *kv_type) {
    if (config && kv_type) {
        *kv_type = config->kv_prim_type();
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_config_set_kv_type(
    nncase::llm::attention_config_node *config, nncase::typecode_t kv_type) {
    if (config) {
        config->kv_prim_type(kv_type);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_create(
    int32_t num_layers, int32_t num_kv_heads, int32_t head_dim,
    nncase::typecode_t kv_type, int32_t block_size,
    const nncase::llm::paged_kvcache_dim_kind *cache_layout,
    const nncase::llm::paged_kvcache_dim_kind *packed_axes,
    int32_t packed_axes_len, const int32_t *lanes, int32_t lanes_len,
    const nncase::llm::paged_kvcache_dim_kind *sharding_axes,
    int32_t sharding_axes_len, const int32_t *axis_policies,
    const int32_t *axis_policies_lens,
    nncase::llm::paged_attention_config_node **config) {
    if (config && cache_layout && packed_axes && lanes && sharding_axes &&
        axis_policies && axis_policies_lens) {
        std::array<nncase::llm::paged_kvcache_dim_kind, 6> cache_layout_arr;
        std::copy(cache_layout, cache_layout + 6, cache_layout_arr.begin());

        std::vector<nncase::llm::paged_kvcache_dim_kind> packed_axes_vec(
            packed_axes, packed_axes + packed_axes_len);

        dims_t lanes_vec(lanes, lanes + lanes_len);

        std::vector<nncase::llm::paged_kvcache_dim_kind> sharding_axes_vec(
            sharding_axes, sharding_axes + sharding_axes_len);

        std::vector<dims_t> axis_policies_vec;
        const int32_t *policy_ptr = axis_policies;
        for (int i = 0; i < sharding_axes_len; i++) {
            dims_t policy(policy_ptr, policy_ptr + axis_policies_lens[i]);
            axis_policies_vec.push_back(policy);
            policy_ptr += axis_policies_lens[i];
        }

        *config = nncase::llm::paged_attention_config(
                      std::in_place, num_layers, num_kv_heads, head_dim,
                      kv_type, block_size, cache_layout_arr, packed_axes_vec,
                      lanes_vec, sharding_axes_vec, axis_policies_vec)
                      .detach();

        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_get_block_size(
    nncase::llm::paged_attention_config_node *config, int32_t *block_size) {
    if (config && block_size) {
        *block_size = config->block_size();
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_set_block_size(
    nncase::llm::paged_attention_config_node *config, int32_t block_size) {
    if (config) {
        config->block_size(block_size);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_get_cache_layout(
    nncase::llm::paged_attention_config_node *config,
    nncase::llm::paged_kvcache_dim_kind *layout, int32_t layout_len) {
    if (config && layout) {
        if (layout_len != 6) {
            return -EINVAL;
        }
        auto src_layout = config->cache_layout();
        std::copy(src_layout.begin(), src_layout.end(), layout);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_set_cache_layout(
    nncase::llm::paged_attention_config_node *config,
    const nncase::llm::paged_kvcache_dim_kind *layout, int32_t layout_len) {
    if (config && layout) {
        if (layout_len != 6) {
            return -EINVAL;
        }
        std::array<nncase::llm::paged_kvcache_dim_kind, 6> cache_layout;
        std::copy(layout, layout + 6, cache_layout.begin());
        config->cache_layout(cache_layout);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_get_packed_axes(
    nncase::llm::paged_attention_config_node *config,
    nncase::llm::paged_kvcache_dim_kind *packed_axes, int32_t packed_axes_len) {
    if (config && packed_axes && packed_axes_len) {
        auto src_axes = config->packed_axes();
        if (packed_axes_len < src_axes.size()) {
            return -EOVERFLOW;
        }
        std::copy(src_axes.begin(), src_axes.end(), packed_axes);
        std::fill_n(packed_axes + src_axes.size(),
                    packed_axes_len - src_axes.size(),
                    (nncase::llm::paged_kvcache_dim_kind)-1);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_set_packed_axes(
    nncase::llm::paged_attention_config_node *config,
    const nncase::llm::paged_kvcache_dim_kind *packed_axes,
    int32_t packed_axes_len) {
    if (config && packed_axes) {
        if (packed_axes_len > 6) {
            return -EOVERFLOW;
        }
        std::vector<nncase::llm::paged_kvcache_dim_kind> axes_vec(
            packed_axes, packed_axes + packed_axes_len);
        config->packed_axes(axes_vec);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_get_lanes(
    nncase::llm::paged_attention_config_node *config, int32_t *lanes,
    int32_t lanes_len) {
    if (config && lanes && lanes_len) {
        auto src_lanes = config->lanes();
        auto required_length = src_lanes.size();
        if (lanes_len < required_length) {
            return -EOVERFLOW;
        }
        std::copy(src_lanes.begin(), src_lanes.end(), lanes);
        std::fill_n(lanes + required_length, lanes_len - required_length, -1);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_set_lanes(
    nncase::llm::paged_attention_config_node *config, const int32_t *lanes,
    int32_t lanes_len) {
    if (config && lanes) {
        if (lanes_len > 8) {
            return -EOVERFLOW;
        }
        config->lanes({lanes, lanes + lanes_len});
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_get_sharding_axes(
    nncase::llm::paged_attention_config_node *config,
    nncase::llm::paged_kvcache_dim_kind *sharding_axes,
    int32_t sharding_axes_len) {
    if (config && sharding_axes && sharding_axes_len) {
        auto src_axes = config->sharding_axes();
        if (sharding_axes_len < src_axes.size()) {
            return -EOVERFLOW;
        }
        std::copy(src_axes.begin(), src_axes.end(), sharding_axes);
        std::fill_n(sharding_axes + src_axes.size(),
                    sharding_axes_len - src_axes.size(),
                    (nncase::llm::paged_kvcache_dim_kind)-1);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_set_sharding_axes(
    nncase::llm::paged_attention_config_node *config,
    const nncase::llm::paged_kvcache_dim_kind *sharding_axes,
    int32_t sharding_axes_len) {
    if (config && sharding_axes) {
        if (sharding_axes_len > 8) {
            return -EOVERFLOW;
        }
        std::vector<nncase::llm::paged_kvcache_dim_kind> axes_vec(
            sharding_axes, sharding_axes + sharding_axes_len);
        config->sharding_axes(axes_vec);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_get_axis_policy_len(
    nncase::llm::paged_attention_config_node *config, int32_t i,
    int32_t *policy_len) {
    if (config && policy_len) {
        auto policies = config->axis_policies();
        if (i >= policies.size()) {
            return -EINVAL;
        }

        auto &policy = policies[i];
        *policy_len = policy.size();
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_get_axis_policy(
    nncase::llm::paged_attention_config_node *config, int32_t i,
    int32_t *axis_policy, int32_t axis_policy_len) {
    if (config && axis_policy) {
        auto policies = config->axis_policies();
        if (i >= policies.size()) {
            return -EINVAL;
        }

        auto &policy = policies[i];
        if (axis_policy_len < policy.size()) {
            return -EOVERFLOW;
        }

        std::copy(policy.begin(), policy.end(), axis_policy);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_config_set_axis_policy(
    nncase::llm::paged_attention_config_node *config, int32_t i,
    const int32_t *axis_policy, int32_t axis_policy_len) {
    if (config && axis_policy) {
        auto policies = config->axis_policies();
        if (i >= policies.size()) {
            return -EINVAL;
        }

        auto policy = dims_t(axis_policy, axis_policy + axis_policy_len);
        config->axis_policies(i, policy);
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_kv_cache_create(
    nncase::llm::attention_config_node *config, int32_t num_seqs,
    int32_t num_tokens, nncase::tensor_node *context_lens,
    nncase::tensor_node *seq_lens,
    nncase::llm::attention_kv_cache_node **cache) {
    if (config && context_lens && seq_lens && cache) {
        *cache =
            nncase::llm::attention_kv_cache(std::in_place, config, num_seqs,
                                            num_tokens, context_lens, seq_lens)
                .detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_kv_cache_get_config(
    nncase::llm::attention_kv_cache_node *cache,
    nncase::llm::attention_config_node **config) {
    if (cache && config) {
        *config = llm::attention_config(cache->config()).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_kv_cache_get_num_seqs(
    nncase::llm::attention_kv_cache_node *cache, int32_t *num_seqs) {
    if (cache && num_seqs) {
        *num_seqs = cache->num_seqs();
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_kv_cache_set_num_seqs(
    nncase::llm::attention_kv_cache_node *cache, int32_t num_seqs) {
    if (cache) {
        cache->num_seqs(num_seqs);
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_kv_cache_get_num_tokens(
    nncase::llm::attention_kv_cache_node *cache, int32_t *num_tokens) {
    if (cache && num_tokens) {
        *num_tokens = cache->num_tokens();
        return 0;
    }
    return -EINVAL;
}

int nncase_attention_kv_cache_set_num_tokens(
    nncase::llm::attention_kv_cache_node *cache, int32_t num_tokens) {
    if (cache) {
        cache->num_tokens(num_tokens);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_kv_cache_create(
    nncase::llm::paged_attention_config_node *config, int32_t num_seqs,
    int32_t num_tokens, nncase::tensor_node *context_lens,
    nncase::tensor_node *seq_lens, nncase::tensor_node *block_table,
    nncase::tensor_node *slot_mapping, int32_t num_blocks,
    const int32_t *kv_shape, int32_t kv_shape_len,
    nncase::llm::paged_attention_kv_cache_node **cache) {
    if (config && context_lens && seq_lens && block_table && slot_mapping &&
        kv_shape && cache) {
        *cache = nncase::llm::paged_attention_kv_cache(
                     std::in_place, config, num_seqs, num_tokens, context_lens,
                     seq_lens, block_table, slot_mapping, num_blocks,
                     dims_t{kv_shape, kv_shape + kv_shape_len})
                     .detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_kv_cache_get_num_blocks(
    nncase::llm::paged_attention_kv_cache_node *cache, int32_t *num_blocks) {
    if (cache && num_blocks) {
        *num_blocks = cache->num_blocks();
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_kv_cache_get_block_table(
    nncase::llm::paged_attention_kv_cache_node *cache,
    nncase::tensor_node **block_table) {
    if (cache && block_table) {
        *block_table = tensor(cache->block_table()).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_kv_cache_set_block_table(
    nncase::llm::paged_attention_kv_cache_node *cache,
    nncase::tensor_node *block_table) {
    if (cache && block_table) {
        cache->block_table(block_table);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_kv_cache_get_slot_mapping(
    nncase::llm::paged_attention_kv_cache_node *cache,
    nncase::tensor_node **slot_mapping) {
    if (cache && slot_mapping) {
        *slot_mapping = tensor(cache->slot_mapping()).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_kv_cache_set_slot_mapping(
    nncase::llm::paged_attention_kv_cache_node *cache,
    nncase::tensor_node *slot_mapping) {
    if (cache && slot_mapping) {
        cache->slot_mapping(slot_mapping);
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_kv_cache_get_kv_cache(
    nncase::llm::paged_attention_kv_cache_node *cache, const int32_t *indices,
    int32_t indices_len, nncase::tensor_node **kv_cache) {
    if (cache && indices && kv_cache) {
        dims_t idx(indices, indices + indices_len);
        *kv_cache = tensor(cache->kv_cache(idx)).detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_paged_attention_kv_cache_set_kv_cache(
    nncase::llm::paged_attention_kv_cache_node *cache, const int32_t *indices,
    int32_t indices_len, nncase::tensor_node *kv_cache) {
    if (cache && indices && kv_cache) {
        dims_t idx(indices, indices + indices_len);
        cache->kv_cache(idx, kv_cache);
        return 0;
    }
    return -EINVAL;
}

int nncase_wait_for_debugger(uint8_t enable) {
#ifndef _WIN32
    if (enable) {
        g_wait_for_debugger = true;
        pid_t pid = getpid();

        while (g_wait_for_debugger) {
            std::cout << "Process " << pid << " is waiting for debugger. "
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        }
    }
    return 0;
#else
    return -ENOSYS;
#endif
}

int nncase_continue_execution() {
#ifndef _WIN32
    g_wait_for_debugger = false;
    return 0;
#else
    return -ENOSYS;
#endif
}
}
