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
#include "nncase/shape.h"
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
#include <ortki/c_api.h>
#include <random>
#include <string>

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
        default: {
        }
        }
    }

    virtual void init_tensor_pow_f32(runtime::runtime_tensor &tensor) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-6.0f, 6.0f);
        NNCASE_UNUSED auto res = kernels::stackvm::apply(
            tensor.shape(), [&](const dims_t &index) -> result<void> {
                get<float>(tensor, index) = static_cast<int32_t>(dis(gen));
                return ok();
            });
    }

    ortki::OrtKITensor *
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

    result<void> check_tuple_output(runtime::runtime_tensor expected,
                                    value_t output) {
        try_var(output_tuple, output.as<tuple>());
        for (size_t i = 0; i < output_tuple->fields().size(); i++) {
            try_var(output_tensor, output_tuple->fields()[i].as<tensor>());
            try_var(output_span,
                    nncase::runtime::get_output_span(output_tensor));
            auto output1 =
                runtime::hrt::create(
                    dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(output_span.data()), 8},
                    true, runtime::host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
            EXPECT_TRUE(is_same_tensor(expected, output1));
        }

        return ok();
    }

    bool is_same_tensor(runtime::runtime_tensor &lhs,
                        runtime::runtime_tensor &rhs) {
        if (lhs.shape() != rhs.shape()) {
            return false;
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
                       case dt_float32: {
                           if (get<float>(lhs, index) ==
                                   get<float>(rhs, index) ||
                               fabs(get<float>(lhs, index) -
                                    get<float>(rhs, index)) < 0.0001f) {
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
