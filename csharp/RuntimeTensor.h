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

#include "stdprefix.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/runtime_tensor_impl.h>
#include <nncase/version.h>
#include <type_traits>
#include <vector>

using namespace nncase;
using namespace nncase::runtime;

template <typename T> datatype_t from_dtype() {
    if (std::is_same_v<T, uint8_t>)
        return dt_uint8;
    else if (std::is_same_v<T, uint16_t>)
        return dt_uint16;
    else if (std::is_same_v<T, uint32_t>)
        return dt_uint32;
    else if (std::is_same_v<T, uint64_t>)
        return dt_uint64;
    else if (std::is_same_v<T, int8_t>)
        return dt_int8;
    else if (std::is_same_v<T, int16_t>)
        return dt_int16;
    else if (std::is_same_v<T, int32_t>)
        return dt_int32;
    else if (std::is_same_v<T, int64_t>)
        return dt_int64;
    else if (std::is_same_v<T, std::bfloat16>)
        throw std::runtime_error("Unsupported float16");
    else if (std::is_same_v<T, float>)
        return dt_float32;
    else if (std::is_same_v<T, double>)
        return dt_float64;
    throw std::runtime_error("Unsupported dtype");
}

inline runtime_shape_t to_shape(const int *shape_ptr, int shape_size) {
    runtime_shape_t shape(shape_size);
    for (size_t i = 0; i < shape.size(); i++)
        shape[i] = (size_t)shape_ptr[i];
    return shape;
}

inline runtime_shape_t to_strides(const int *stride_ptr, int stride_size) {
    runtime_shape_t strides(stride_size);
    for (size_t i = 0; i < strides.size(); i++)
        strides[i] = (size_t)stride_ptr[i];

    return strides;
}

/**
 * @brief create tensor form buffer
 *
 * @param buffer_ptr the buffer ptr type is dtype
 * @param datatype the datatype enum value.
 * @param shape_ptr  the shape pointer
 * @param shape_size the shape array size
 * @param total_items the total elements counts
 * @param item_size  the each element total bytes
 * @param stride_ptr the stide pointer( by element)
 * @return void*
 */
EXPORT_API(runtime_tensor *)
RuntimeTensor_from_buffer(const uint8_t *buffer_ptr, datatype_t datatype,
                          const int *shape_ptr, int shape_size,
                          size_t total_items, size_t item_size,
                          const int *stride_ptr) {
    auto hostrt =
        host_runtime_tensor::create(
            (datatype_t)datatype, to_shape(shape_ptr, shape_size),
            to_strides(stride_ptr, shape_size),
            gsl::make_span((gsl::byte *)(buffer_ptr), total_items * item_size),
            [=](gsl::byte *) {})
            .unwrap_or_throw();
    auto rt = new runtime_tensor(std::move(hostrt));
    return rt;
}

EXPORT_API(void)
RuntimeTensor_free(runtime_tensor *rt) { delete rt; }

EXPORT_API(void)
RuntimeTensor_to_buffer(runtime_tensor *rt, uint8_t *buffer_ptr,
                        uint8_t *datatype_ptr) {
    if (!rt->is_contiguous()) {
        std::cout << "not_supported uncontiguous tensor!" << std::endl;
    }
    *datatype_ptr = rt->datatype();

    auto host = rt->as_host().unwrap_or_throw();
    auto src_map = std::move(hrt::map(host, hrt::map_read).unwrap_or_throw());
    auto src_buffer = src_map.buffer();
    memcpy(buffer_ptr, src_buffer.data(), src_buffer.size_bytes());
}

/**
 * @brief copy the runtime Tensor
 *
 * @param from
 * @param to
 */
EXPORT_API(void)
RuntimeTensor_copy_to(runtime_tensor *from, runtime_tensor *dest) {
    from->copy_to(*dest).unwrap_or_throw();
}

/**
 * @brief get the runtime tensor datatype enum value.
 *
 * @return uint8_t
 */
EXPORT_API(datatype_t)
RuntimeTensor_dtype(runtime_tensor *rt) { return rt->datatype(); }

/**
 * @brief get the shape, if shape_ptr is null, just return
 *
 * @param rt
 * @param shape_ptr
 * @return int
 */
EXPORT_API(int)
RuntimeTensor_shape(runtime_tensor *rt, int *shape_ptr) {
    auto rt_shape = rt->shape();
    if (shape_ptr != nullptr) {
        for (size_t i = 0; i < rt_shape.size(); i++) {
            shape_ptr[i] = rt_shape[i];
        }
    }
    return rt_shape.size();
}

/**
 * @brief get the strides, if stride_ptr is null, just return
 *
 * @param rt
 * @param shape_ptr
 * @return int
 */
EXPORT_API(int)
RuntimeTensor_strides(runtime_tensor *rt, int *strides_ptr) {
    auto strides = rt->strides();
    if (strides_ptr != nullptr) {
        for (size_t i = 0; i < strides.size(); i++) {
            strides_ptr[i] = strides[i];
        }
    }
    return strides.size();
}
