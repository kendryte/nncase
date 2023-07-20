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
#include "macro_util.h"
#include "nncase/shape.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/kernels/apply.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/util.h>
#include <numeric>
#include <ortki/c_api.h>
#include <random>
#include <string>
#include <vector>

using namespace nncase::runtime;
using namespace nncase::kernels;
namespace nncase {
class KernelTest {
  public:
    template <typename T>
    T &get(runtime::runtime_tensor &t, gsl::span<const size_t> index) {
        auto map = std::move(
            runtime::hrt::map(t, runtime::map_read).unwrap_or_throw());
        auto data = map.buffer().as_span<T>();
        return data[kernels::offset(t.strides(), index)];
    }

    virtual void init_tensor(runtime::runtime_tensor &tensor) {
        auto dtype = tensor.datatype();
        switch (dtype) {
        case dt_int8: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-6, 6);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int8_t>(tensor, index) = static_cast<int8_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_int16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-6, 6);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int16_t>(tensor, index) =
                        static_cast<int16_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_int32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-6, 6);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int32_t>(tensor, index) = dis(gen);
                    return ok();
                });
            break;
        }
        case dt_int64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-6, 6);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int64_t>(tensor, index) =
                        static_cast<int64_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint8: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 127);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint8_t>(tensor, index) =
                        static_cast<uint8_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 127);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint16_t>(tensor, index) =
                        static_cast<uint16_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 127);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint32_t>(tensor, index) =
                        static_cast<uint32_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, 127);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint64_t>(tensor, index) =
                        static_cast<uint64_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<half>(tensor, index) = static_cast<half>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<float>(tensor, index) = static_cast<float>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<double>(tensor, index) = static_cast<double>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_boolean: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<bool>(tensor, index) =
                        static_cast<double>(dis(gen)) >= 0;
                    return ok();
                });
            break;
        }
        default: {
        }
        }
    }

    void cast_copy_tensor(runtime::runtime_tensor &source_tensor,
                          runtime::runtime_tensor &destination_tensor) {
        auto destination_tensor_dtype = destination_tensor.datatype();
        auto source_tensor_dtype = source_tensor.datatype();
        switch (destination_tensor_dtype) {
        case dt_int8: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<int8_t>(destination_tensor, index) =
                            static_cast<int8_t>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<int8_t>(destination_tensor, index) =
                            static_cast<int8_t>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<int8_t>(destination_tensor, index) =
                            static_cast<int8_t>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<int8_t>(destination_tensor, index) =
                            static_cast<int8_t>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<int8_t>(destination_tensor, index) =
                            static_cast<int8_t>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<int8_t>(destination_tensor, index) =
                            static_cast<int8_t>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<int8_t>(destination_tensor, index) =
                            static_cast<int8_t>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<int8_t>(destination_tensor, index) =
                            static_cast<int8_t>(
                                get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<int8_t>(destination_tensor, index) =
                            static_cast<int8_t>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        case dt_int16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<int16_t>(destination_tensor, index) =
                            static_cast<int16_t>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<int16_t>(destination_tensor, index) =
                            static_cast<int16_t>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<int16_t>(destination_tensor, index) =
                            static_cast<int16_t>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<int16_t>(destination_tensor, index) =
                            static_cast<int16_t>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<int16_t>(destination_tensor, index) =
                            static_cast<int16_t>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<int16_t>(destination_tensor, index) =
                            static_cast<int16_t>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<int16_t>(destination_tensor, index) =
                            static_cast<int16_t>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<int16_t>(destination_tensor, index) =
                            static_cast<int16_t>(
                                get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<int16_t>(destination_tensor, index) =
                            static_cast<int16_t>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        case dt_int32: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<int32_t>(destination_tensor, index) =
                            static_cast<int32_t>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<int32_t>(destination_tensor, index) =
                            static_cast<int32_t>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<int32_t>(destination_tensor, index) =
                            static_cast<int32_t>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<int32_t>(destination_tensor, index) =
                            static_cast<int32_t>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<int32_t>(destination_tensor, index) =
                            static_cast<int32_t>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<int32_t>(destination_tensor, index) =
                            static_cast<int32_t>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<int32_t>(destination_tensor, index) =
                            static_cast<int32_t>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<int32_t>(destination_tensor, index) =
                            static_cast<int32_t>(
                                get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<int32_t>(destination_tensor, index) =
                            static_cast<int32_t>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        case dt_int64: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<int64_t>(destination_tensor, index) =
                            static_cast<int64_t>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<int64_t>(destination_tensor, index) =
                            static_cast<int64_t>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<int64_t>(destination_tensor, index) =
                            static_cast<int64_t>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<int64_t>(destination_tensor, index) =
                            static_cast<int64_t>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<int64_t>(destination_tensor, index) =
                            static_cast<int64_t>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<int64_t>(destination_tensor, index) =
                            static_cast<int64_t>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<int64_t>(destination_tensor, index) =
                            static_cast<int64_t>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<int64_t>(destination_tensor, index) =
                            static_cast<int64_t>(
                                get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<int64_t>(destination_tensor, index) =
                            static_cast<int64_t>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        case dt_uint8: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<uint8_t>(destination_tensor, index) =
                            static_cast<uint8_t>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<uint8_t>(destination_tensor, index) =
                            static_cast<uint8_t>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<uint8_t>(destination_tensor, index) =
                            static_cast<uint8_t>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<uint8_t>(destination_tensor, index) =
                            static_cast<uint8_t>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<uint8_t>(destination_tensor, index) =
                            static_cast<uint8_t>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<uint8_t>(destination_tensor, index) =
                            static_cast<uint8_t>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<uint8_t>(destination_tensor, index) =
                            static_cast<uint8_t>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<uint8_t>(destination_tensor, index) =
                            static_cast<uint8_t>(
                                get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<uint8_t>(destination_tensor, index) =
                            static_cast<uint8_t>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        case dt_uint16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<uint16_t>(destination_tensor, index) =
                            static_cast<uint16_t>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<uint16_t>(destination_tensor, index) =
                            static_cast<uint16_t>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<uint16_t>(destination_tensor, index) =
                            static_cast<uint16_t>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<uint16_t>(destination_tensor, index) =
                            static_cast<uint16_t>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<uint16_t>(destination_tensor, index) =
                            static_cast<uint16_t>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<uint16_t>(destination_tensor, index) =
                            static_cast<uint16_t>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<uint16_t>(destination_tensor, index) =
                            static_cast<uint16_t>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<uint16_t>(destination_tensor, index) =
                            static_cast<uint16_t>(
                                get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<uint16_t>(destination_tensor, index) =
                            static_cast<uint16_t>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        case dt_uint32: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<uint32_t>(destination_tensor, index) =
                            static_cast<uint32_t>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<uint32_t>(destination_tensor, index) =
                            static_cast<uint32_t>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<uint32_t>(destination_tensor, index) =
                            static_cast<uint32_t>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<uint32_t>(destination_tensor, index) =
                            static_cast<uint32_t>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<uint32_t>(destination_tensor, index) =
                            static_cast<uint32_t>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<uint32_t>(destination_tensor, index) =
                            static_cast<uint32_t>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<uint32_t>(destination_tensor, index) =
                            static_cast<uint32_t>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<uint32_t>(destination_tensor, index) =
                            static_cast<uint32_t>(
                                get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<uint32_t>(destination_tensor, index) =
                            static_cast<uint32_t>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        case dt_uint64: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<uint64_t>(destination_tensor, index) =
                            static_cast<uint64_t>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<uint64_t>(destination_tensor, index) =
                            static_cast<uint64_t>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<uint64_t>(destination_tensor, index) =
                            static_cast<uint64_t>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<uint64_t>(destination_tensor, index) =
                            static_cast<uint64_t>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<uint64_t>(destination_tensor, index) =
                            static_cast<uint64_t>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<uint64_t>(destination_tensor, index) =
                            static_cast<uint64_t>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<uint64_t>(destination_tensor, index) =
                            static_cast<uint64_t>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<uint64_t>(destination_tensor, index) =
                            static_cast<uint64_t>(
                                get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<uint64_t>(destination_tensor, index) =
                            static_cast<uint64_t>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        case dt_float32: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<float>(destination_tensor, index) =
                            static_cast<float>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<float>(destination_tensor, index) =
                            static_cast<float>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<float>(destination_tensor, index) =
                            static_cast<float>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<float>(destination_tensor, index) =
                            static_cast<float>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<float>(destination_tensor, index) =
                            static_cast<float>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<float>(destination_tensor, index) =
                            static_cast<float>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<float>(destination_tensor, index) =
                            static_cast<float>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<float>(destination_tensor, index) =
                            static_cast<float>(get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<float>(destination_tensor, index) =
                            static_cast<float>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        case dt_float64: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                destination_tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    switch (source_tensor_dtype) {
                    case dt_int8: {
                        get<double>(destination_tensor, index) =
                            static_cast<double>(
                                get<int8_t>(source_tensor, index));
                        break;
                    }
                    case dt_int16: {
                        get<double>(destination_tensor, index) =
                            static_cast<double>(
                                get<int16_t>(source_tensor, index));
                        break;
                    }
                    case dt_int32: {
                        get<double>(destination_tensor, index) =
                            static_cast<double>(
                                get<int32_t>(source_tensor, index));
                        break;
                    }
                    case dt_int64: {
                        get<double>(destination_tensor, index) =
                            static_cast<double>(
                                get<int64_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint16: {
                        get<double>(destination_tensor, index) =
                            static_cast<double>(
                                get<uint16_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint32: {
                        get<double>(destination_tensor, index) =
                            static_cast<double>(
                                get<uint32_t>(source_tensor, index));
                        break;
                    }
                    case dt_uint64: {
                        get<double>(destination_tensor, index) =
                            static_cast<double>(
                                get<uint64_t>(source_tensor, index));
                        break;
                    }
                    case dt_float16: {
                        get<double>(destination_tensor, index) =
                            static_cast<double>(
                                get<half>(source_tensor, index));
                        break;
                    }
                    case dt_float32: {
                        get<double>(destination_tensor, index) =
                            static_cast<double>(
                                get<float>(source_tensor, index));
                        break;
                    }
                    default: {
                    }
                    }
                    return ok();
                });
            break;
        }
        default: {
        }
        }
    }

    static ortki::OrtKITensor *
    runtime_tensor_2_ort_tensor(runtime::runtime_tensor &tensor) {
        auto mapped =
            std::move(runtime::hrt::map(tensor, runtime::map_read).unwrap());
        void *buffer = reinterpret_cast<void *>(mapped.buffer().data());

        ortki::DataType ort_type = ortki::DataType_FLOAT;
        auto dtype = tensor.datatype();
        switch (dtype) {
        case dt_boolean: {
            ort_type = ortki::DataType_BOOL;
            break;
        }
        case dt_int8: {
            ort_type = ortki::DataType_INT8;
            break;
        }
        case dt_int16: {
            ort_type = ortki::DataType_INT16;
            break;
        }
        case dt_int32: {
            ort_type = ortki::DataType_INT32;
            break;
        }
        case dt_int64: {
            ort_type = ortki::DataType_INT64;
            break;
        }
        case dt_uint8: {
            ort_type = ortki::DataType_UINT8;
            break;
        }
        case dt_uint16: {
            ort_type = ortki::DataType_UINT16;
            break;
        }
        case dt_uint32: {
            ort_type = ortki::DataType_UINT32;
            break;
        }
        case dt_uint64: {
            ort_type = ortki::DataType_UINT64;
            break;
        }
        case dt_float16: {
            ort_type = ortki::DataType_FLOAT16;
            break;
        }
        case dt_float32: {
            ort_type = ortki::DataType_FLOAT;
            break;
        }
        case dt_float64: {
            ort_type = ortki::DataType_DOUBLE;
            break;
        }
        default: {
            std::cerr << "unsupported data type: dtype = " << dtype
                      << std::endl;
            std::abort();
        }
        }

        const int64_t *shape =
            reinterpret_cast<const int64_t *>(tensor.shape().data());
        auto shape_size = tensor.shape().size();
        return make_tensor(buffer, ort_type, shape, shape_size);
    }

    // static void * runtime_tensor_2_vector_type(runtime::runtime_tensor
    // &tensor) {
    //     // auto mapped =
    //     //     std::move(runtime::hrt::map(tensor,
    //     runtime::map_read).unwrap());
    //     // void *buffer = reinterpret_cast<void *>(mapped.buffer().data());
    //     // return buffer;
    //     void* arr;
    //     auto dtype = tensor.datatype();
    //     switch (dtype) {
    //         NNCASE_CONDITION(int8)
    //         NNCASE_CONDITION(int16)
    //         NNCASE_CONDITION(int32)
    //         NNCASE_CONDITION(int64)
    //         NNCASE_CONDITION(uint8)
    //         NNCASE_CONDITION(uint16)
    //         NNCASE_CONDITION(uint32)
    //         NNCASE_CONDITION(uint64)
    //         case dt_float32:
    //             arr = new float[tensor.shape().size()];
    //             break;
    //         case dt_float64:
    //             arr = new double[tensor.shape().size()];
    //             break;
    //         case dt_float16:
    //             arr = new half[tensor.shape().size()];
    //             break;
    //         case dt_boolean:
    //             arr = new bool[tensor.shape().size()];
    //             break;
    //     default:
    //         break;
    //     }
    //     kernels::stackvm::apply(
    //         tensor.shape(),
    //         [&](gsl::span<const size_t> index) -> result<void> {
    //             auto dtype = tensor.datatype();
    //             switch (dtype) {
    //                 NNCASE_CONDITION_GET(int8)
    //                 NNCASE_CONDITION_GET(int16)
    //                 NNCASE_CONDITION_GET(int32)
    //                 NNCASE_CONDITION_GET(int64)
    //                 NNCASE_CONDITION_GET(uint8)
    //                 NNCASE_CONDITION_GET(uint16)
    //                 NNCASE_CONDITION_GET(uint32)
    //                 NNCASE_CONDITION_GET(uint64)
    //                 case dt_float32:
    //                     arr[index] = static_cast<float>(get<float>(tensor,
    //                     index)); break;
    //                 case dt_float64:
    //                     arr[index] = static_cast<double>(get<double>(tensor,
    //                     index)); break;
    //                 case dt_float16:
    //                     arr[index] = static_cast<double>(get<half>(tensor,
    //                     index)); break;
    //                 case dt_boolean:
    //                     arr[index] = static_cast<bool>(get<bool>(tensor,
    //                     index)); break;
    //                 default:
    //                     break;
    //             }
    //             return ok();
    //         }).is_ok();
    //     return arr;

    // }

    result<void> check_tuple_output(runtime::runtime_tensor expected[],
                                    typecode_t dtypes[],
                                    const value_t &output) {
        try_var(output_tuple, output.as<tuple>());
        for (size_t i = 0; i < output_tuple->fields().size(); i++) {
            try_var(output_tensor, output_tuple->fields()[i].as<tensor>());
            try_var(output_span,
                    nncase::runtime::get_output_span(output_tensor));
            auto output1 =
                runtime::hrt::create(
                    dtypes[i], expected[i].shape(),
                    {reinterpret_cast<gsl::byte *>(output_span.data()),
                     output_span.size_bytes()},
                    true, runtime::host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
            bool result = is_same_tensor(expected[i], output1) ||
                          cosine_similarity_tensor(expected[i], output1);
            if (!result) {
                std::cout << "expected ";
                print_runtime_tensor(expected[i]);
                std::cout << "actual ";
                print_runtime_tensor(output1);
            }
            EXPECT_TRUE(result);
        }

        return ok();
    }

    bool is_same_tensor(runtime::runtime_tensor &lhs,
                        runtime::runtime_tensor &rhs) {
        if (lhs.shape() != rhs.shape()) {
            if (rhs.shape().size() != 0 || lhs.shape().size() != 1 ||
                lhs.shape()[0] != 1) {
                return false;
            }
        }

        return kernels::stackvm::apply(
                   lhs.shape(),
                   [&](gsl::span<const size_t> index) -> result<void> {
                       auto dtype = lhs.datatype();
                       switch (dtype) {
                       case dt_int8: {
                           if (get<int8_t>(lhs, index) ==
                               get<int8_t>(rhs, index)) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_int16: {
                           if (get<int16_t>(lhs, index) ==
                               get<int16_t>(rhs, index)) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_int32: {
                           if (get<int32_t>(lhs, index) ==
                               get<int32_t>(rhs, index)) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_int64: {
                           if (get<int64_t>(lhs, index) ==
                               get<int64_t>(rhs, index)) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_uint8: {
                           if (get<uint8_t>(lhs, index) ==
                               get<uint8_t>(rhs, index)) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_uint16: {
                           if (get<uint16_t>(lhs, index) ==
                               get<uint16_t>(rhs, index)) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_uint32: {
                           if (get<uint32_t>(lhs, index) ==
                               get<uint32_t>(rhs, index)) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_uint64: {
                           if (get<uint64_t>(lhs, index) ==
                               get<uint64_t>(rhs, index)) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_float16: {
                           if (get<half>(lhs, index) ==
                                   get<half>(rhs, index) ||
                               fabs(get<half>(lhs, index) -
                                    get<half>(rhs, index)) <=
                                   std::numeric_limits<float>::epsilon()) {
                               return ok();
                           } else if (std::isnan(get<half>(lhs, index)) &&
                                      std::isnan(get<half>(rhs, index))) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_float32: {
                           if (get<float>(lhs, index) ==
                                   get<float>(rhs, index) ||
                               fabs(get<float>(lhs, index) -
                                    get<float>(rhs, index)) <=
                                   std::numeric_limits<float>::epsilon()) {
                               return ok();
                           } else if (std::isnan(get<float>(lhs, index)) &&
                                      std::isnan(get<float>(rhs, index))) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_float64: {
                           if (get<double>(lhs, index) ==
                               get<double>(rhs, index)) {
                               return ok();
                           } else if (std::isnan(get<double>(lhs, index)) &&
                                      std::isnan(get<double>(rhs, index))) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_boolean: {
                           if (get<bool>(lhs, index) == get<bool>(rhs, index)) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       default: {
                           return err(std::errc::not_supported);
                       }
                       }
                   })
            .is_ok();
    }

    bool cosine_similarity_tensor(runtime::runtime_tensor &lhs,
                                  runtime::runtime_tensor &rhs) {
        if (lhs.shape() != rhs.shape()) {
            if (rhs.shape().size() != 0 || lhs.shape().size() != 1 ||
                lhs.shape()[0] != 1) {
                return false;
            }
        }

        std::vector<float> vec1;
        std::vector<float> vec2;
        vec1.reserve(compute_size(lhs.shape()));
        vec2.reserve(compute_size(rhs.shape()));

        kernels::stackvm::apply(
            lhs.shape(),
            [&](gsl::span<const size_t> index) -> result<void> {
                auto dtype = lhs.datatype();
                switch (dtype) {
                case dt_int8: {
                    vec1.push_back(static_cast<float>(get<int8_t>(lhs, index)));
                    vec2.push_back(static_cast<float>(get<int8_t>(rhs, index)));
                    break;
                }
                case dt_int16: {
                    vec1.push_back(
                        static_cast<float>(get<int16_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<float>(get<int16_t>(rhs, index)));
                    break;
                }
                case dt_int32: {
                    vec1.push_back(
                        static_cast<float>(get<int32_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<float>(get<int32_t>(rhs, index)));
                    break;
                }
                case dt_int64: {
                    vec1.push_back(
                        static_cast<float>(get<int64_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<float>(get<int64_t>(rhs, index)));
                    break;
                }
                case dt_uint8: {
                    vec1.push_back(
                        static_cast<float>(get<uint8_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<float>(get<uint8_t>(rhs, index)));
                    break;
                }
                case dt_uint16: {
                    vec1.push_back(
                        static_cast<float>(get<uint16_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<float>(get<uint16_t>(rhs, index)));
                    break;
                }
                case dt_uint32: {
                    vec1.push_back(
                        static_cast<float>(get<uint32_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<float>(get<uint32_t>(rhs, index)));
                    break;
                }
                case dt_uint64: {
                    vec1.push_back(
                        static_cast<float>(get<uint64_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<float>(get<uint64_t>(rhs, index)));
                    break;
                }
                case dt_float32: {
                    vec1.push_back(get<float>(lhs, index));
                    vec2.push_back(get<float>(rhs, index));
                    break;
                }
                case dt_float64: {
                    vec1.push_back(static_cast<float>(get<double>(lhs, index)));
                    vec2.push_back(static_cast<float>(get<double>(rhs, index)));
                    break;
                }
                case dt_boolean: {
                    vec1.push_back(
                        static_cast<float>(get<bool>(lhs, index) ? 2 : 1));
                    vec2.push_back(
                        static_cast<float>(get<bool>(rhs, index) ? 2 : 1));
                    break;
                }
                default: {
                    return err(std::errc::not_supported);
                }
                }
                return ok();
            })
            .is_ok();

        float dotProduct =
            std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0f);
        float norm1 = std::sqrt(
            std::inner_product(vec1.begin(), vec1.end(), vec1.begin(), 0.0f));
        float norm2 = std::sqrt(
            std::inner_product(vec2.begin(), vec2.end(), vec2.begin(), 0.0f));
        float cosine_similarity = dotProduct / (norm1 * norm2);

        std::cout << "cosine_similarity:" << cosine_similarity << std::endl;
        return cosine_similarity > 0.99f; // Return true if cosine
                                          // similarity is close to 1
    }

    void print_runtime_tensor(runtime::runtime_tensor lhs) {
        std::cout << "tensor:" << std::endl;
        kernels::stackvm::apply(
            lhs.shape(),
            [&](gsl::span<const size_t> index) -> result<void> {
                auto dtype = lhs.datatype();
                switch (dtype) {
                case dt_int8:
                    std::cout << static_cast<int8_t>(get<int8_t>(lhs, index))
                              << " ";
                    break;
                case dt_int16:
                    std::cout << static_cast<int16_t>(get<int16_t>(lhs, index))
                              << " ";
                    break;
                case dt_int32:
                    std::cout << static_cast<int32_t>(get<int32_t>(lhs, index))
                              << " ";
                    break;
                case dt_int64:
                    std::cout << static_cast<int64_t>(get<int64_t>(lhs, index))
                              << " ";
                    break;
                case dt_uint8:
                    std::cout << static_cast<uint8_t>(get<uint8_t>(lhs, index))
                              << " ";
                    break;
                case dt_uint16:
                    std::cout
                        << static_cast<uint16_t>(get<uint16_t>(lhs, index))
                        << " ";
                    break;
                case dt_uint32:
                    std::cout
                        << static_cast<uint32_t>(get<uint32_t>(lhs, index))
                        << " ";
                    break;
                case dt_uint64:
                    std::cout
                        << static_cast<uint64_t>(get<uint64_t>(lhs, index))
                        << " ";
                    break;
                case dt_float32:
                    std::cout << get<float>(lhs, index) << " ";
                    break;
                case dt_float64:
                    std::cout << static_cast<double>(get<double>(lhs, index))
                              << " ";
                    break;
                case dt_float16:
                    std::cout << static_cast<double>(get<half>(lhs, index))
                              << " ";
                    break;
                case dt_boolean:
                    std::cout << static_cast<bool>(get<bool>(lhs, index))
                              << " ";
                    break;
                default:
                    break;
                }
                return ok();
            })
            .is_ok();

        std::cout << std::endl;
    }

    void ort_tensor_dump(ortki::OrtKITensor *ort) {
        size_t size = tensor_length(ort);
        std::cout << "ort: size = " << size << std::endl;

        size_t rank = tensor_rank(ort);
        std::cout << "ort: rank = " << rank << ":";

        int64_t shape[16] = {0};
        tensor_shape(ort, shape);
        for (size_t i = 0; i < rank; i++)
            std::cout << " " << shape[i];

        std::cout << std::endl;
    }
};
} // namespace nncase
