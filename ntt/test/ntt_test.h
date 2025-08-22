/* Copyright 2019-2024 Canaan Inc.
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
#include "nncase/float8.h"
#include "nncase/half.h"
#include "nncase/ntt/apply.h"
#include "nncase/ntt/ntt.h"
#include "nncase/ntt/shape.h"
#include <assert.h>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <stdio.h>

#ifdef __AVX2__
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#define ULP_SIZE 250000
#ifndef ULP_SIZE
#define ULP_SIZE 10000
#endif

#if defined __x86_64
#ifndef CPU_FREQUENCY_MHZ
#define CPU_FREQUENCY_MHZ 4100
#endif

#ifndef CLOCK_SOURCE_FREQUENCY_MHZ
#define CLOCK_SOURCE_FREQUENCY_MHZ 27
#endif

#elif defined __riscv
#ifndef CPU_FREQUENCY_MHZ
#define CPU_FREQUENCY_MHZ 1600
#endif

#ifndef CLOCK_SOURCE_FREQUENCY_MHZ
#define CLOCK_SOURCE_FREQUENCY_MHZ 27
#endif

#else
#ifndef CPU_FREQUENCY_MHZ
#define CPU_FREQUENCY_MHZ 1000
#endif

#ifndef CLOCK_SOURCE_FREQUENCY_MHZ
#define CLOCK_SOURCE_FREQUENCY_MHZ 27
#endif
#endif

namespace nncase {
namespace NttTest {

__inline__ uint64_t get_cpu_cycle(void) {
    uint64_t cycles = 0;
#if defined __x86_64
    __asm__ __volatile__("" : : : "memory");
    cycles = __rdtsc();
    __asm__ __volatile__("" : : : "memory");
#elif defined __riscv
    uint64_t time = 0;
    asm volatile("rdtime %0" : "=r"(time));
    cycles = time * CPU_FREQUENCY_MHZ / CLOCK_SOURCE_FREQUENCY_MHZ;
#endif
    return cycles;
}

template <ntt::TensorOrVector TTensor>
void print_tensor(TTensor &tensor, std::string name);

template <typename T, TensorOrVector TTensor> 
void generate_random_tensor([[maybe_unused]] TTensor &tensor, [[maybe_unused]] std::mt19937 &gen, [[maybe_unused]] T start = static_cast<T>(0),
                 [[maybe_unused]] T stop = static_cast<T>(1)) {
    std::cerr << __FUNCTION__ << ": unsupported data type" << std::endl;
    std::abort();
}

template <typename T, TensorOrVector TTensor> 
requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
void generate_random_tensor(TTensor &tensor, std::mt19937 &gen, T start = static_cast<T>(0),
                 T stop = static_cast<T>(1), bool allow_zr = true, [[maybe_unused]] bool only_int = true) {
    std::uniform_int_distribution<int64_t> dis(start, stop);
    ntt::apply(tensor.shape(), [&](auto &index) {
        if (allow_zr) {
            tensor(index) = static_cast<T>(dis(gen));
        } else {
            do {
                tensor(index) = static_cast<T>(dis(gen));
                // std::cout << tensor(index) << std::endl;
            } while (tensor(index) == static_cast<T>(0));
        }
    });
}

template <typename T, TensorOrVector TTensor>
requires(std::is_floating_point_v<T>)
void generate_random_tensor(TTensor &tensor, std::mt19937 &gen, T start = static_cast<T>(0),
                            T stop = static_cast<T>(1), bool allow_zr = true, bool only_int = false) {
    
    auto fill_with_distribution = [&](auto &distribution) {
        ntt::apply(tensor.shape(), [&](auto &index) {
            if (allow_zr) {
                tensor(index) = static_cast<T>(distribution(gen));
            } else {
                T value;
                do {
                    value = static_cast<T>(distribution(gen));
                } while (value == static_cast<T>(0));
                tensor(index) = value;
            }
        });
    };

    if (only_int) {
        //bf16 has __bf16 and float cast funtion on x86 which has native bfloat16.
        //directly cast to int64_t would occur ambiguous
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, half>){
            std::uniform_int_distribution<int64_t> dis(static_cast<int64_t>(static_cast<float>(start)), static_cast<int64_t>(static_cast<float>(stop)));
            fill_with_distribution(dis);
        }
        else{
            std::uniform_int_distribution<int64_t> dis(static_cast<int64_t>(start), static_cast<int64_t>(stop));
            fill_with_distribution(dis);
        }
    } else {
        std::uniform_real_distribution<double> dis(start, stop);
        fill_with_distribution(dis);
    }
}

template <typename T, TensorOrVector TTensor>
requires(std::is_same_v<T, bool>)
void generate_random_tensor(TTensor &tensor, std::mt19937 &gen, [[maybe_unused]]T start = static_cast<T>(0),
                            [[maybe_unused]]T stop = static_cast<T>(1), [[maybe_unused]]bool allow_zr = true, [[maybe_unused]]bool only_int = false) {
    std::uniform_int_distribution<int> dis(0, 1);
    ntt::apply(tensor.shape(), [&](auto &index) {
        tensor(index) = static_cast<bool>(dis(gen) < 0.5);
    });
}


template <typename T>
T nextToNeg1(T x) {
    //TODO:  special handling for 0
    // TODO: logic is wrong about negative numbers
    // std::cout << "x:" << x << std::endl;
    float x_f = static_cast<float>(x);
    x_f = std::fabs(x_f);

    // std::cout << "x_f:" << x_f << std::endl;
    static_assert(sizeof(T) == 1 || sizeof(T) == 2,
                  "nextToNeg1 only supports 8-bit or 16-bit formats");
    using int_type = std::conditional_t<sizeof(T) == 1, std::uint8_t, std::uint16_t>;

    T x_abs = static_cast<T>(x_f);
    // std::cout << "x_abs:" << x_abs << std::endl;
    
    int_type x_i = std::bit_cast<int_type>(x_abs);
    x_i = (x_i - 1);  
    T x_lower = std::bit_cast<T>(x_i);
    // std::cout << "x_lower" << x_lower <<std::endl;
    return x_lower;
}


template <typename T> T ulp(T x) {
    // For standard floating point types (float, double, long double)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, long double>) {
        x = std::fabs(x);
        if (std::isfinite(x)) {
            T lower = std::nextafter(x, static_cast<T>(-1.0));
            return x - lower;
        }
        return x;
    } else {
        // For custom floating point types (half, bfloat16, etc.)
        // Convert to float for ULP computation

        if (!std::isfinite((float)x)) {
            return x;
        }
        //if(x == 0) //TODO
        T x_abs = (T)std::fabs((float)x);
        T lower = nextToNeg1(x_abs);
        // printf("ulp: %f of %f\n", (float)(x_abs - lower), (float)x);
        return x_abs - lower;
    }
}

template <typename T>
bool are_close(T a, T b, double abs_tol = 1e-6,  double rel_tol = 1e-5) {
    // The short-circuit for equality is important for performance and to handle infinities.
    if (a == b) {
        return true;
    }
    
    // ULP check for all non-integer types (including float, half, double, etc.)
    if constexpr (!std::is_integral_v<T>) {
        // std::cout << "std::fabs(a-b) " << std::fabs((a-b))  <<std::endl;
        // std::cout << "ulp(b):" <<ulp(b) << "   ulp(a)" << ulp(a) << std::endl;

        if (std::fabs(double(a - b)) <= double(ulp(b)) || std::fabs(double(a - b)) <= double(ulp(a))) {
            return true;
        }
    }
    
    // Special handling for float type: if a is float_max_from_exp and b is greater than float_max_from_exp, return true
    if constexpr (std::is_same_v<T, float>) {
        const T float_max_from_exp = 1.65164e+38f;
        // Using relative tolerance for floating-point comparison to handle precision issues
        if (std::abs(a - float_max_from_exp) <= std::max(abs_tol, rel_tol * std::max(std::abs(a), std::abs(float_max_from_exp))) && b > float_max_from_exp) {
            return true;
        } 
    }


    return std::abs(double(a - b)) <= std::max(abs_tol, rel_tol * std::max(std::abs(double(a)), std::abs(double(b))));
}

template <typename T, TensorOrVector TTensor> 
requires(std::is_same_v<T, bool>)
void generate_random_tensor(TTensor &tensor, std::mt19937 &gen, [[maybe_unused]] T start = static_cast<T>(0),
                 [[maybe_unused]] T stop = static_cast<T>(1), [[maybe_unused]] bool allow_zr = true) {
    std::uniform_int_distribution<int> dis(0, 1);
    ntt::apply(tensor.shape(), [&](auto &index) {
        tensor(index) = static_cast<bool>(dis(gen));
    });
}

template <typename T, TensorOrVector TTensor>
void init_tensor(TTensor &tensor, T start = static_cast<T>(0),
                 T stop = static_cast<T>(1), bool allow_zr = true, [[maybe_unused]] bool only_int = false) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // } else if constexpr (std::is_same_v<T, bool>) {
    //     std::uniform_real_distribution<double> dis(0.0, 1.0);
    //     ntt::apply(tensor.shape(), [&](auto &index) {
    //         tensor(index) = static_cast<double>(dis(gen)) >= 0.5;
    //     });
    generate_random_tensor(tensor, gen, start, stop, allow_zr, only_int);
}

template <typename T, TensorOfVector TTensor>
void init_tensor(TTensor &tensor, T start = static_cast<T>(0),
                 T stop = static_cast<T>(1), bool allow_zr = true, bool only_int = false) {
    ntt::apply(tensor.shape(),
               [&](auto &index) { init_tensor(tensor(index), start, stop, allow_zr, only_int); });
}

inline double calculate_cosine_similarity(const std::vector<double>& v1, const std::vector<double>& v2) {

    double dotProduct = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
    double norm1 = std::sqrt(std::inner_product(v1.begin(), v1.end(), v1.begin(), 0.0));
    double norm2 = std::sqrt(std::inner_product(v2.begin(), v2.end(), v2.begin(), 0.0));
    std::cout << "dotProduct: " << dotProduct << ", norm1: " << norm1
              << ", norm2: " << norm2 << std::endl;
    return dotProduct / (norm1 * norm2);
}

template <ntt::TensorOrVector TTensor1, ntt::TensorOrVector TTensor2>
bool compare_tensor(TTensor1 &lhs, TTensor2 &rhs, double threshold = 0.999f) {
    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<double> v1;
    std::vector<double> v2;
    v1.reserve(lhs.shape().length());
    v2.reserve(rhs.shape().length());

    bool pass = true;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        auto lvalue = lhs(index);
        auto rvalue = rhs(index);
        auto d1 = static_cast<double>(
            (lhs(index)));
        auto d2 = static_cast<double>(
            (rhs(index)));
        v1.push_back(d1);
        v2.push_back(d2);
        if (!are_close(lvalue, rvalue)) {
            // #ifndef NDEBUG
            std::cout << "index = (";
            for (size_t i = 0; i < index.rank(); i++)
                std::cout << index[i] << " ";
            std::cout << "): lhs = " << d1 << ", rhs = " << d2 << std::endl;
            // #endif
            pass = false;
        }
    });

    if (!pass) {
        double dotProduct = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
        std::cout << "dotProduct" << dotProduct << std::endl;
        double norm1 = std::sqrt(std::inner_product(v1.begin(), v1.end(), v1.begin(), 0.0));
        std::cout << "norm1" << norm1 << std::endl;
        double norm2 = std::sqrt(std::inner_product(v2.begin(), v2.end(), v2.begin(), 0.0));
        std::cout << "norm2" << norm2 << std::endl;
        double cosine_similarity = calculate_cosine_similarity(v1, v2);
        pass = cosine_similarity > threshold;
        if (!pass)
            std::cerr << "cosine_similarity = " << cosine_similarity << std::endl;
    }
    return pass;
}

template <ntt::TensorOfVector TTensor1, ntt::TensorOfVector TTensor2>
    requires(TTensor1::element_type::rank() == 1)
bool compare_tensor(TTensor1 &lhs, TTensor2 &rhs, double threshold = 0.999f) {
    using vector_type = typename TTensor1::element_type;
    constexpr size_t N = vector_type::template lane<0>();
    printf("N = %zu\n", N);
    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<double> v1;
    std::vector<double> v2;
    v1.reserve(lhs.shape().length() * N);
    v2.reserve(rhs.shape().length() * N);

    bool pass = true;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        const auto lvalue = lhs(index);
        const auto rvalue = rhs(index);

        nncase::ntt::apply(lvalue.shape(), [&](auto idx) {
            auto d1 = static_cast<double>(
                static_cast<typename decltype(lvalue)::element_type>(
                    lvalue(idx)));
            auto d2 = static_cast<double>(
                static_cast<typename decltype(rvalue)::element_type>(
                    rvalue(idx)));
            // auto d1 = int32_t(lvalue(idx));
            // auto d2 = int32_t(rvalue(idx));
            v1.push_back(d1);
            v2.push_back(d2);
            if (!are_close(d1, d2)) {
                // #ifndef NDEBUG
                std::cout << "index = (";
                for (size_t i = 0; i < index.rank(); i++)
                    std::cout << index[i] << " ";
                std::cout << "): lhs = " << d1 << ", rhs = " << d2 << std::endl;
                // #endif
                pass = false;
            }
        });
    });

    if (!pass) {
        double cosine_similarity = calculate_cosine_similarity(v1, v2);
        pass = cosine_similarity > threshold;
        if (!pass)
            std::cerr << "cosine_similarity = " << cosine_similarity
                      << std::endl;
    }
    return pass;
}

// 2D vector
template <ntt::TensorOfVector TTensor1, ntt::TensorOfVector TTensor2>
    requires(TTensor1::element_type::rank() == 2 &&
             TTensor2::element_type::rank() == 2)
bool compare_tensor(TTensor1 &lhs, TTensor2 &rhs, double threshold = 0.999f) {
    using vector_type = typename TTensor1::element_type;
    constexpr size_t N0 = vector_type::template lane<0>();
    constexpr size_t N1 = vector_type::template lane<1>();

    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<double> v1;
    std::vector<double> v2;
    v1.reserve(lhs.shape().length() * N0 * N1);
    v2.reserve(rhs.shape().length() * N0 * N1);

    bool pass = true;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        const auto lvalue = lhs(index);
        const auto rvalue = rhs(index);

        nncase::ntt::apply(lvalue.shape(), [&](auto idx) {
            auto d1 = static_cast<double>(
                static_cast<typename decltype(lvalue)::element_type>(
                    lvalue(idx)));
            auto d2 = static_cast<double>(
                static_cast<typename decltype(rvalue)::element_type>(
                    rvalue(idx)));
            v1.push_back(d1);
            v2.push_back(d2);
            if (!are_close(d1, d2)) {
                // #ifndef NDEBUG
                std::cout << "index = (";
                for (size_t i = 0; i < index.rank(); i++)
                    std::cout << index[i] << " ";
                std::cout << "): lhs = " << d1 << ", rhs = " << d2 << std::endl;
                // #endif
                pass = false;
            }
        });
    });

    if (!pass) {
        double cosine_similarity = calculate_cosine_similarity(v1, v2);
        pass = cosine_similarity > threshold;
        if (!pass)
            std::cerr << "cosine_similarity = " << cosine_similarity
                      << std::endl;
    }
    return pass;
}

template <ntt::TensorOrVector TTensor>
void print_tensor(TTensor &tensor, std::string name) {
    printf("%s\n", name.c_str() );
    using element_type = typename TTensor::element_type;
    if constexpr (ntt::Vector<element_type>) {
        nncase::ntt::apply(tensor.shape(), [&](auto index) {
            print_tensor(tensor(index), name + "[" +
                                       std::to_string(index[0]) + "]");

        });
    } else {
        nncase::ntt::apply(tensor.shape(), [&](auto index) {
            auto value = tensor(index);
            using value_type = decltype(value);
            if constexpr (std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>) {
                printf("%ld ", static_cast<int64_t>(value));
            } else {
                if constexpr (requires { typename decltype(value)::element_type; }) {
                    // value is a proxy, extract the element
                    auto act_val = static_cast<typename decltype(value)::element_type>(value);
                    printf("%lf ", static_cast<double>(act_val));
                } else {
                    // value is already the actual type
                    printf("%lf ", static_cast<double>(value));
                }
            }
        });
    }

    printf("\n");
}

template <ntt::TensorOrVector TTensor_src, ntt::TensorOrVector TTensor_dst>
void reinterpret_cast_fp8_to_uint8(const TTensor_src &tensor_src,
                                   TTensor_dst &tensor_dst) {
    using element_type = typename TTensor_src::element_type;
    if constexpr (ntt::Vector<element_type>) {
        nncase::ntt::apply(tensor_src.shape(), [&](auto index) {
            auto vec_src = tensor_src(index);
            auto &vec_dst = tensor_dst(index);
            nncase::ntt::apply(vec_src.shape(), [&](auto idx) {
                vec_dst(idx) = std::bit_cast<uint8_t>(vec_src(idx).raw());
            });
        });
    } else {
        nncase::ntt::apply(tensor_src.shape(), [&](auto index) {
            auto vec_src = tensor_src(index);
            auto &vec_dst = tensor_dst(index);
            vec_dst = std::bit_cast<uint8_t>(vec_src.raw());
        });
    }
}
// 1D vecvtor

// template <typename T, typename Shape, typename Stride, size_t N>
// void print_tensor(ntt::tensor<ntt::vector<T, N>, Shape, Stride> &lhs,
//                   std::string name) {
//     std::cout << name << std::endl;

//     nncase::ntt::apply(lhs.shape(), [&](auto index) {
//         const ntt::vector<T, N> vec = lhs(index);

//         nncase::ntt::apply(vec.shape(), [&](auto idx) {
//             auto d1 = (double)(vec(idx));
//             std::cout << d1 << " ";
//         });
//     });

//     std::cout << std::endl;
// }


template <typename T, typename Shape, typename Stride>
bool compare_ulp(ntt::tensor<T, Shape, Stride> &lhs,
                 ntt::tensor<T, Shape, Stride> &rhs, double threshold = 0.5f) {
    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<T> v1;
    std::vector<T> v2;
    v1.reserve(lhs.shape().length());
    v2.reserve(rhs.shape().length());

    bool pass = true;
    double max_ulp_error = 0.f;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        auto lvalue = lhs(index);
        auto rvalue = rhs(index);
        auto ulp_error = std::abs(lvalue - rvalue) / ulp(rvalue);
        max_ulp_error = ulp_error > max_ulp_error ? ulp_error : max_ulp_error;
    });

    if (max_ulp_error > threshold) {
        std::cout << "ulp threshold = " << threshold
                  << ", max_ulp_error = " << max_ulp_error << std::endl;

        pass = false;
    }
    // std::cout << "ulp threshold = " << threshold
    //           << ", max_ulp_error = " << max_ulp_error << std::endl;
    return pass;
}

template <typename T, size_t N, typename Shape, typename Stride>
bool compare_ulp(ntt::tensor<ntt::vector<T, N>, Shape, Stride> &lhs,
                 ntt::tensor<ntt::vector<T, N>, Shape, Stride> &rhs,
                 double threshold = 0.5f) {
    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<T> v1;
    std::vector<T> v2;
    v1.reserve(lhs.shape().length());
    v2.reserve(rhs.shape().length());

    bool pass = true;
    double max_ulp_error = 0.f;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        auto lvalue = lhs(index);
        auto rvalue = rhs(index);

        nncase::ntt::apply(lvalue.shape(), [&](auto idx) {
            auto ulp_error =
                std::abs(lvalue(idx) - rvalue(idx)) / ulp((T)rvalue(idx));
            if (ulp_error > max_ulp_error)
                std::cout << "lvalue(idx) = " << lvalue(idx)
                          << ", rvalue(idx) = " << rvalue(idx)
                          << ", ulp = " << ulp((T)rvalue(idx))
                          << ", ulp_error = " << ulp_error
                          << ", max_ulp_error = " << max_ulp_error << std::endl;
            max_ulp_error =
                ulp_error > max_ulp_error ? ulp_error : max_ulp_error;
        });
    });

    if (max_ulp_error > threshold) {
        std::cout << "ulp threshold = " << threshold
                  << ", max_ulp_error = " << max_ulp_error << std::endl;

        pass = false;
    }
    std::cout << "ulp threshold = " << threshold
              << ", max_ulp_error = " << max_ulp_error << std::endl;
    return pass;
}

} // namespace NttTest
} // namespace nncase