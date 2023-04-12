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

NNCASE_API int nncase_dtype_get_typecode(nncase::datatype_node *dtype);

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
}
