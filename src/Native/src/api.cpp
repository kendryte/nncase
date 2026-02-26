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
#include <nncase/object.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>

using namespace nncase;
using namespace nncase::runtime;

namespace {
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
        node->release();
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
            {reinterpret_cast<const gsl::byte *>(model_buffer), model_size},
            copy_buffer));
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
        gsl::span<value_t> param_values{reinterpret_cast<value_t *>(params),
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
                              nncase::datatype_node **dtype) {
    if (dtype) {
        c_try_var(type, datatype_t::from_typecode(typecode));
        *dtype = type.detach();
        return 0;
    }
    return -EINVAL;
}

int nncase_dtype_get_typecode(nncase::datatype_node *dtype) {
    return dtype->typecode();
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
}
