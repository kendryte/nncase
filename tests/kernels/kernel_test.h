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
#include "generated/generated_macro.h"
#include "macro_util.h"
#include "nncase/shape.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
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
#include <rapidjson/document.h> // rapidjson's DOM-style API
#include <rapidjson/error/en.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <string>
#include <vector>

using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace rapidjson;
namespace nncase {
typedef enum { RANDOM, NOZERO, NONEG, NOPOS } initial_mode;

class KernelTest {
  public:
    template <typename T>
    T &get(runtime::runtime_tensor &t, std::span<const size_t> index) {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
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
                [&](std::span<const size_t> index) -> result<void> {
                    get<bool>(tensor, index) =
                        static_cast<double>(dis(gen)) >= 0;
                    return ok();
                });
            break;
        }
        case dt_bfloat16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<bfloat16>(tensor, index) =
                        static_cast<bfloat16>(dis(gen));
                    return ok();
                });
            break;
        }
        default: {
        }
        }
    }

    template <typename T> T InitAttributeSCALAR(dims_t shape, T initvalue) {
        if (shape.size() == 1 && (shape[0] == 1)) {
            // Scalar attribute
            return initvalue;
        } else {
            std::cout << "Shape is error, not a Scalar\n";
            return 0;
        }
    }

    template <typename T>
    T *InitAttributeARRAYONEDIM(dims_t shape, std::vector<T> initvalue) {
        if (shape.size() == 1 && (shape[0] == initvalue.size())) {
            // One dim array attribute
            T *tmp = new T[shape[0]];
            for (size_t i = 0; i < shape[0]; ++i) {
                tmp[i] = initvalue[i];
            }
            return tmp;
        } else {
            std::cout << "Shape is error, not a array with one dim\n";
            return nullptr;
        }
    }

    void InitTensor(runtime::runtime_tensor &tensor, initial_mode mode) {
        auto dtype = tensor.datatype();
        std::uniform_int_distribution<> int_random_dis(-6, 6);
        std::uniform_int_distribution<> uint_random_dis(0, 6);
        std::uniform_int_distribution<> int_noneg_dis(0, 6);
        std::uniform_int_distribution<> int_nopos_dis(-6, 0);

        std::uniform_real_distribution<> real_random_dis(-6, 6);
        std::uniform_real_distribution<> real_noneg_dis(0, 6);
        std::uniform_real_distribution<> real_nopos_dis(-6, 0);

        std::bernoulli_distribution bool_dis(0.5);

        switch (dtype) {
        case dt_int8: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-6, 6);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(uint8_t, int_random_dis, int_noneg_dis,
                                     int_noneg_dis, int_nopos_dis)
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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(int16_t, int_random_dis, int_noneg_dis,
                                     int_noneg_dis, int_nopos_dis)
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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(int32_t, int_random_dis, int_noneg_dis,
                                     int_noneg_dis, int_nopos_dis)

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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(int64_t, int_random_dis, int_noneg_dis,
                                     int_noneg_dis, int_nopos_dis)

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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(uint8_t, uint_random_dis, int_noneg_dis,
                                     int_noneg_dis, uint_random_dis)
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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(uint16_t, uint_random_dis, int_noneg_dis,
                                     int_noneg_dis, uint_random_dis)

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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(uint32_t, uint_random_dis, int_noneg_dis,
                                     int_noneg_dis, uint_random_dis)

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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(uint64_t, uint_random_dis, int_noneg_dis,
                                     int_noneg_dis, uint_random_dis)
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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(half, real_random_dis, real_noneg_dis,
                                     real_noneg_dis, real_nopos_dis)
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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(float, real_random_dis, real_noneg_dis,
                                     real_noneg_dis, real_nopos_dis)
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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(double, real_random_dis, real_noneg_dis,
                                     real_noneg_dis, real_nopos_dis)
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
                [&](std::span<const size_t> index) -> result<void> {
                    SWITCH_INIT_MODE(bool, bool_dis, bool_dis, bool_dis,
                                     bool_dis)
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
        case dt_bfloat16: {
            ort_type = ortki::DataType_BFLOAT16;
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
    //         [&](std::span<const size_t> index) -> result<void> {
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
                    {reinterpret_cast<std::byte *>(output_span.data()),
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

    template <typename T>
    std::vector<T> tensor_to_array(runtime::runtime_tensor &lhs) {

        std::vector<T> vec1;
        vec1.reserve(compute_size(lhs.shape()));

        kernels::stackvm::apply(
            lhs.shape(),
            [&](std::span<const size_t> index) -> result<void> {
                auto dtype = lhs.datatype();
                switch (dtype) {
                case dt_int8: {
                    vec1.push_back(static_cast<T>(get<int8_t>(lhs, index)));
                    break;
                }
                case dt_int16: {
                    vec1.push_back(static_cast<T>(get<int16_t>(lhs, index)));
                    break;
                }
                case dt_int32: {
                    vec1.push_back(static_cast<T>(get<int32_t>(lhs, index)));
                    break;
                }
                case dt_int64: {
                    vec1.push_back(static_cast<T>(get<int64_t>(lhs, index)));
                    break;
                }
                case dt_uint8: {
                    vec1.push_back(static_cast<T>(get<uint8_t>(lhs, index)));
                    break;
                }
                case dt_uint16: {
                    vec1.push_back(static_cast<T>(get<uint16_t>(lhs, index)));
                    break;
                }
                case dt_uint32: {
                    vec1.push_back(static_cast<T>(get<uint32_t>(lhs, index)));
                    break;
                }
                case dt_uint64: {
                    vec1.push_back(static_cast<T>(get<uint64_t>(lhs, index)));
                    break;
                }
                case dt_float16: {
                    vec1.push_back(static_cast<T>(get<half>(lhs, index)));
                    break;
                }
                case dt_bfloat16: {
                    vec1.push_back(static_cast<T>(get<bfloat16>(lhs, index)));
                    break;
                }
                case dt_float32: {
                    vec1.push_back(static_cast<T>(get<float>(lhs, index)));
                    break;
                }
                case dt_float64: {
                    vec1.push_back(static_cast<T>(get<double>(lhs, index)));
                    break;
                }
                default: {
                    return err(std::errc::not_supported);
                }
                }
                return ok();
            })
            .is_ok();

        return vec1;
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
                   [&](std::span<const size_t> index) -> result<void> {
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
                           if (get<half>(lhs, index) == get<half>(rhs, index) ||
                               fabs((float)get<half>(lhs, index) -
                                    (float)get<half>(rhs, index)) <= 0.01f) {
                               return ok();
                           } else if (std::isnan(get<half>(lhs, index)) &&
                                      std::isnan(get<half>(rhs, index))) {
                               return ok();
                           } else {
                               return err(std::errc::not_supported);
                           }
                           break;
                       }
                       case dt_bfloat16: {
                           if (get<bfloat16>(lhs, index) ==
                                   get<bfloat16>(rhs, index) ||
                               fabs(get<bfloat16>(lhs, index) -
                                    get<bfloat16>(rhs, index)) <=
                                   std::numeric_limits<float>::epsilon()) {
                               return ok();
                           } else if (std::isnan(get<bfloat16>(lhs, index)) &&
                                      std::isnan(get<bfloat16>(rhs, index))) {
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
                                    get<float>(rhs, index)) <= 0.0001f
                               /*std::numeric_limits<float>::epsilon()*/) {
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

        std::vector<double> vec1;
        std::vector<double> vec2;
        vec1.reserve(compute_size(lhs.shape()));
        vec2.reserve(compute_size(rhs.shape()));

        kernels::stackvm::apply(
            lhs.shape(),
            [&](std::span<const size_t> index) -> result<void> {
                auto dtype = lhs.datatype();
                switch (dtype) {
                case dt_int8: {
                    vec1.push_back(
                        static_cast<double>(get<int8_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<int8_t>(rhs, index)));
                    break;
                }
                case dt_int16: {
                    vec1.push_back(
                        static_cast<double>(get<int16_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<int16_t>(rhs, index)));
                    break;
                }
                case dt_int32: {
                    vec1.push_back(
                        static_cast<double>(get<int32_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<int32_t>(rhs, index)));
                    break;
                }
                case dt_int64: {
                    vec1.push_back(
                        static_cast<double>(get<int64_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<int64_t>(rhs, index)));
                    break;
                }
                case dt_uint8: {
                    vec1.push_back(
                        static_cast<double>(get<uint8_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<uint8_t>(rhs, index)));
                    break;
                }
                case dt_uint16: {
                    vec1.push_back(
                        static_cast<double>(get<uint16_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<uint16_t>(rhs, index)));
                    break;
                }
                case dt_uint32: {
                    vec1.push_back(
                        static_cast<double>(get<uint32_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<uint32_t>(rhs, index)));
                    break;
                }
                case dt_uint64: {
                    vec1.push_back(
                        static_cast<double>(get<uint64_t>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<uint64_t>(rhs, index)));
                    break;
                }
                case dt_float16: {
                    vec1.push_back(static_cast<double>(get<half>(lhs, index)));
                    vec2.push_back(static_cast<double>(get<half>(rhs, index)));
                    break;
                }
                case dt_bfloat16: {
                    vec1.push_back(
                        static_cast<double>(get<bfloat16>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<bfloat16>(rhs, index)));
                    break;
                }
                case dt_float32: {
                    vec1.push_back(static_cast<double>(get<float>(lhs, index)));
                    vec2.push_back(static_cast<double>(get<float>(rhs, index)));
                    break;
                }
                case dt_float64: {
                    vec1.push_back(
                        static_cast<double>(get<double>(lhs, index)));
                    vec2.push_back(
                        static_cast<double>(get<double>(rhs, index)));
                    break;
                }
                default: {
                    return err(std::errc::not_supported);
                }
                }
                return ok();
            })
            .is_ok();

        double dotProduct = std::inner_product(vec1.begin(), vec1.end(),
                                               vec2.begin(), (double)0.0);
        double norm1 = std::sqrt(std::inner_product(vec1.begin(), vec1.end(),
                                                    vec1.begin(), (double)0.0));
        double norm2 = std::sqrt(std::inner_product(vec2.begin(), vec2.end(),
                                                    vec2.begin(), (double)0.0));
        double cosine_similarity = dotProduct / (norm1 * norm2);

        std::cout << "cosine_similarity:" << cosine_similarity << std::endl;

        // Return true if cosine similarity is close to 1
        return cosine_similarity > 0.99f;
    }

    void print_runtime_tensor(runtime::runtime_tensor lhs) {
        std::cout << "tensor (shape:[ ";
        for (auto a : lhs.shape())
            std::cout << a << " ";
        std::cout << "]):" << std::endl;
        kernels::stackvm::apply(
            lhs.shape(),
            [&](std::span<const size_t> index) -> result<void> {
                auto dtype = lhs.datatype();
                switch (dtype) {
                case dt_int8:
                    std::cout << get<int8_t>(lhs, index) << " ";
                    break;
                case dt_int16:
                    std::cout << get<int16_t>(lhs, index) << " ";
                    break;
                case dt_int32:
                    std::cout << get<int32_t>(lhs, index) << " ";
                    break;
                case dt_int64:
                    std::cout << get<int64_t>(lhs, index) << " ";
                    break;
                case dt_uint8:
                    std::cout << get<uint8_t>(lhs, index) << " ";
                    break;
                case dt_uint16:
                    std::cout << get<uint16_t>(lhs, index) << " ";
                    break;
                case dt_uint32:
                    std::cout << get<uint32_t>(lhs, index) << " ";
                    break;
                case dt_uint64:
                    std::cout << get<uint64_t>(lhs, index) << " ";
                    break;
                case dt_float32:
                    std::cout << get<float>(lhs, index) << " ";
                    break;
                case dt_float64:
                    std::cout << get<double>(lhs, index) << " ";
                    break;
                case dt_float16:
                    std::cout << get<half>(lhs, index) << " ";
                    break;
                case dt_boolean:
                    std::cout << get<bool>(lhs, index) << " ";
                    break;
                case dt_bfloat16:
                    std::cout << get<bfloat16>(lhs, index) << " ";
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

    virtual void quantize_to_int16(runtime::runtime_tensor &expected,
                                   runtime::runtime_tensor &input, int16_t zero,
                                   float scale) {
        if (expected.datatype() != dt_int16)
            return;
        NNCASE_UNUSED auto res = kernels::stackvm::apply(
            expected.shape(),
            [&](std::span<const size_t> index) -> result<void> {
                get<int16_t>(expected, index) = static_cast<int16_t>(
                    get<float>(input, index) / scale + zero);
                return ok();
            });
    }

    virtual void int16_dequantize_to_float(runtime::runtime_tensor &expected,
                                           runtime::runtime_tensor &input,
                                           int16_t zero, float scale) {
        if (input.datatype() != dt_int16)
            return;
        NNCASE_UNUSED auto res = kernels::stackvm::apply(
            expected.shape(),
            [&](std::span<const size_t> index) -> result<void> {
                get<float>(expected, index) = static_cast<float>(
                    (get<int16_t>(input, index) - zero) * scale);
                return ok();
            });
    }

    static std::string ReadFromJsonFile(std::ifstream &file) {
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        file.close();
        return content;
    }

    static void ParseJson(Document &document, std::string js_str) {
        if (document.Parse<kParseCommentsFlag>(js_str.c_str()).HasParseError())
            std::cout << "Parsing Error: "
                      << (unsigned)document.GetErrorOffset() << " "
                      << GetParseError_En(document.GetParseError())
                      << std::endl;

        if (!document.IsObject()) {
            throw std::runtime_error("type error! it should be Object.");
        }
    }

    void ParseJson(std::string js_str) {
        if (_document.Parse<kParseCommentsFlag>(js_str.c_str()).HasParseError())
            std::cout << "Parsing Error: "
                      << (unsigned)_document.GetErrorOffset() << " "
                      << GetParseError_En(_document.GetParseError())
                      << std::endl;

        if (!_document.IsObject()) {
            throw std::runtime_error("type error! it should be Object.");
        }
    }

    typecode_t Str2DataType(std::string type) {
        std::cout << type << std::endl;
        if (str_2_datatype.find(type) != str_2_datatype.end()) {
            return str_2_datatype[type];
        } else {
            return dt_int8;
        }
    }

    int64_t GetNumber(const char *key) {
        if (!_document[key].IsInt64()) {
            throw std::runtime_error("type error! it should be int64.");
        }

        return _document[key].GetInt64();
    }

    float GetFloatNumber(const char *key) {
        if (!_document[key].IsDouble()) {
            throw std::runtime_error("type error! it should be double.");
        }

        return _document[key].GetFloat();
    }

    typecode_t GetDataType(const char *key) {
        if (!_document[key].IsString()) {
            throw std::runtime_error("type error! it should be string.");
        }

        return Str2DataType(_document[key].GetString());
    }

    std::string GetString(const char *key) {
        if (!_document[key].IsString()) {
            throw std::runtime_error("type error! it should be string.");
        }

        return _document[key].GetString();
    }

    dims_t GetShapeArray(const char *key) {
        if (!_document[key].IsArray()) {
            throw std::runtime_error("type error! it should be array.");
        }

        Value &array = _document[key];
        size_t arraySize = array.Size();
        dims_t cArray(arraySize);
        for (rapidjson::SizeType i = 0; i < arraySize; i++) {
            if (array[i].IsInt()) {
                cArray[i] = array[i].GetInt();
            } else {
                std::cout << "Invalid JSON format. Expected unsigned integer "
                             "values in the array."
                          << std::endl;
            }
        }
        return cArray;
    }

    std::vector<int64_t> GetDataArray(const char *key) {
        if (!_document[key].IsArray()) {
            throw std::runtime_error("type error! it should be array.");
        }

        Value &array = _document[key];
        size_t arraySize = array.Size();
        std::vector<int64_t> cArray(arraySize);
        for (rapidjson::SizeType i = 0; i < arraySize; i++) {
            if (array[i].IsInt()) {
                cArray[i] = array[i].GetInt();
            } else {
                std::cout << "Invalid JSON format. Expected unsigned integer "
                             "values in the array."
                          << std::endl;
            }
        }
        return cArray;
    }

    axes_t GetAxesArray(const char *key) {
        if (!_document[key].IsArray()) {
            throw std::runtime_error("type error! it should be array.");
        }

        Value &array = _document[key];
        size_t arraySize = array.Size();
        axes_t cArray(arraySize);
        for (rapidjson::SizeType i = 0; i < arraySize; i++) {
            if (array[i].IsInt()) {
                cArray[i] = array[i].GetInt();
            } else {
                std::cout << "Invalid JSON format. Expected unsigned integer "
                             "values in the array."
                          << std::endl;
            }
        }
        return cArray;
    }

    static std::string GetFileNameFromMacro(const char *filePath) {
        std::string fullFilePath(filePath);
        size_t lastSlashIndex = fullFilePath.find_last_of("/\\");
        if (lastSlashIndex != std::string::npos) {
            return fullFilePath.substr(lastSlashIndex + 1);
        }
        return fullFilePath;
    }

  public:
    Document _document;
    std::map<std::string, typecode_t> str_2_datatype = {
        {"dt_int8", dt_int8},       {"dt_int16", dt_int16},
        {"dt_int32", dt_int32},     {"dt_int64", dt_int64},
        {"dt_uint8", dt_uint8},     {"dt_uint16", dt_uint16},
        {"dt_uint32", dt_uint32},   {"dt_uint64", dt_uint64},
        {"dt_float16", dt_float16}, {"dt_float32", dt_float32},
        {"dt_float64", dt_float64}, {"dt_bfloat16", dt_bfloat16},
        {"dt_boolean", dt_boolean}};
};
} // namespace nncase
