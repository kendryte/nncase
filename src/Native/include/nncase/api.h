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
#include <nncase/compiler_defs.h>
#include <nncase/runtime/simple_types.h>

namespace nncase {
class object_node;
class tensor_node;
class tuple_node;
class value_node;
class type_node;
class datatype_node;
class vector_type_node;
namespace llm {
class attention_config_node;
class paged_attention_config_node;
class attention_kv_cache_node;
class paged_attention_kv_cache_node;
class paged_attention_scheduler_node;
enum class paged_attention_dim_kind;
enum class attention_cache_kind;
} // namespace llm

namespace runtime {
class buffer_allocator;
class buffer_node;
class host_buffer_node;
class interpreter;
class runtime_function;
} // namespace runtime
} // namespace nncase

extern "C" {
struct nncase_buffer_slice {
    nncase::runtime::buffer_node *buffer;
    uint32_t start;
    uint32_t size_bytes;
};

NNCASE_API int nncase_object_add_ref(nncase::object_node *node);
NNCASE_API int nncase_object_release(nncase::object_node *node);

NNCASE_API int nncase_interp_create(nncase::runtime::interpreter **interp);
NNCASE_API int nncase_interp_free(nncase::runtime::interpreter *interp);
NNCASE_API int nncase_interp_load_model(nncase::runtime::interpreter *interp,
                                        void *model_buffer, uint32_t model_size,
                                        bool copy_buffer);
NNCASE_API int
nncase_interp_load_model_from_path(nncase::runtime::interpreter *interp,
                                   const char *model_path);
NNCASE_API int nncase_interp_set_dump_root(nncase::runtime::interpreter *interp,
                                           const char *path);
NNCASE_API int
nncase_interp_get_entry_func(nncase::runtime::interpreter *interp,
                             nncase::runtime::runtime_function **func);

NNCASE_API int
nncase_func_get_params_size(nncase::runtime::runtime_function *func,
                            uint32_t *size);
NNCASE_API int nncase_func_invoke(nncase::runtime::runtime_function *func,
                                  nncase::value_node **params,
                                  uint32_t params_size,
                                  nncase::value_node **result);

NNCASE_API int
nncase_buffer_allocator_get_host(nncase::runtime::buffer_allocator **alloc);
NNCASE_API int
nncase_buffer_allocator_alloc(nncase::runtime::buffer_allocator *alloc,
                              uint32_t bytes, void *options,
                              nncase::runtime::buffer_node **buffer);
NNCASE_API int
nncase_buffer_as_host(nncase::runtime::buffer_node *buffer,
                      nncase::runtime::host_buffer_node **host_buffer);

NNCASE_API int
nncase_host_buffer_map(nncase::runtime::host_buffer_node *host_buffer,
                       nncase::runtime::map_access_t access, void **data,
                       uint32_t *bytes);
NNCASE_API int
nncase_host_buffer_unmap(nncase::runtime::host_buffer_node *host_buffer);

NNCASE_API int nncase_dtype_create_prime(nncase::typecode_t typecode,
                                         nncase::datatype_node **dtype);

NNCASE_API int nncase_dtype_create_vector(nncase::datatype_node *elem_type,
                                          int32_t *lanes, int32_t length,
                                          nncase::datatype_node **dtype);

NNCASE_API int nncase_dtype_get_typecode(nncase::datatype_node *dtype);

NNCASE_API int
nncase_vector_dtype_get_elem_type(nncase::vector_type_node *handle,
                                  nncase::datatype_node **elemType);

NNCASE_API int
nncase_vector_dtype_get_lanes_length(nncase::vector_type_node *handle,
                                     int32_t *length);

NNCASE_API int nncase_vector_dtype_get_lanes(nncase::vector_type_node *handle,
                                             int32_t *lanes);

NNCASE_API int nncase_value_is_tensor(nncase::value_node *value,
                                      bool *is_tensor);

NNCASE_API int nncase_tensor_create(nncase::datatype_node *dtype,
                                    const uint32_t *dims, uint32_t dims_length,
                                    const uint32_t *strides,
                                    uint32_t strides_length,
                                    nncase_buffer_slice *buffer,
                                    nncase::tensor_node **tensor);
NNCASE_API int nncase_tensor_get_dtype(nncase::tensor_node *tensor,
                                       nncase::datatype_node **dtype);
NNCASE_API int nncase_tensor_get_buffer(nncase::tensor_node *tensor,
                                        nncase_buffer_slice *buffer);
NNCASE_API int nncase_tensor_get_dims(nncase::tensor_node *tensor,
                                      uint32_t *dims, uint32_t *dims_length);
NNCASE_API int nncase_tensor_get_strides(nncase::tensor_node *tensor,
                                         uint32_t *dims, uint32_t *dims_length);

NNCASE_API int nncase_tuple_create(nncase::value_node **fields,
                                   uint32_t fields_length,
                                   nncase::tuple_node **tuple);
NNCASE_API int nncase_tuple_get_fields(nncase::tuple_node *tuple,
                                       nncase::value_node **fields,
                                       uint32_t *fields_length);
NNCASE_API int
nncase_attention_config_create(int32_t num_layers, int32_t num_kv_heads,
                               int32_t head_dim, nncase::typecode_t kv_type,
                               nncase::llm::attention_config_node **config);

NNCASE_API int nncase_attention_config_get_num_layers(
    nncase::llm::attention_config_node *config, int32_t *num_layers);

NNCASE_API int nncase_attention_config_set_num_layers(
    nncase::llm::attention_config_node *config, int32_t num_layers);

NNCASE_API int nncase_attention_config_get_num_kv_heads(
    nncase::llm::attention_config_node *config, int32_t *num_kv_heads);

NNCASE_API int nncase_attention_config_set_num_kv_heads(
    nncase::llm::attention_config_node *config, int32_t num_kv_heads);

NNCASE_API int
nncase_attention_config_get_head_dim(nncase::llm::attention_config_node *config,
                                     int32_t *head_dim);

NNCASE_API int
nncase_attention_config_set_head_dim(nncase::llm::attention_config_node *config,
                                     int32_t head_dim);

NNCASE_API int
nncase_attention_config_get_kv_type(nncase::llm::attention_config_node *config,
                                    nncase::typecode_t *kv_type);

NNCASE_API int
nncase_attention_config_set_kv_type(nncase::llm::attention_config_node *config,
                                    nncase::typecode_t kv_type);

NNCASE_API int nncase_paged_attention_config_create(
    int32_t num_layers, int32_t num_kv_heads, int32_t head_dim,
    nncase::typecode_t kv_type, int32_t block_size,
    const nncase::llm::paged_attention_dim_kind *cache_layout,
    const nncase::llm::paged_attention_dim_kind *packed_axes,
    int32_t packed_axes_len, const int32_t *lanes, int32_t lanes_len,
    const int32_t *topology, int32_t topology_len,
    nncase::llm::paged_attention_config_node **config);

NNCASE_API int nncase_paged_attention_config_get_block_size(
    nncase::llm::paged_attention_config_node *config, int32_t *block_size);

NNCASE_API int nncase_paged_attention_config_set_block_size(
    nncase::llm::paged_attention_config_node *config, int32_t block_size);

NNCASE_API int nncase_paged_attention_config_get_cache_layout(
    nncase::llm::paged_attention_config_node *config,
    nncase::llm::paged_attention_dim_kind *layout, int32_t layout_len);

NNCASE_API int nncase_paged_attention_config_set_cache_layout(
    nncase::llm::paged_attention_config_node *config,
    const nncase::llm::paged_attention_dim_kind *layout, int32_t layout_len);

NNCASE_API int nncase_paged_attention_config_get_packed_axes(
    nncase::llm::paged_attention_config_node *config,
    nncase::llm::paged_attention_dim_kind *packed_axes,
    int32_t packed_axes_len);

NNCASE_API int nncase_paged_attention_config_set_packed_axes(
    nncase::llm::paged_attention_config_node *config,
    const nncase::llm::paged_attention_dim_kind *packed_axes,
    int32_t packed_axes_len);

NNCASE_API int nncase_paged_attention_config_get_lanes(
    nncase::llm::paged_attention_config_node *config, int32_t *lanes,
    int32_t lanes_len);

NNCASE_API int nncase_paged_attention_config_set_lanes(
    nncase::llm::paged_attention_config_node *config, const int32_t *lanes,
    int32_t lanes_len);

NNCASE_API int nncase_paged_attention_config_get_topology(
    nncase::llm::paged_attention_config_node *config, int32_t *topology,
    int32_t topology_len);

NNCASE_API int nncase_paged_attention_config_set_topology(
    nncase::llm::paged_attention_config_node *config, const int32_t *topology,
    int32_t topology_len);
}
