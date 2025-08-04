#include "ntt_test.h"
#include <iomanip>
#include <iostream>
#include <map>
#include <nncase/ntt/ntt.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <type_traits>
#include <utility>

using namespace nncase;

bool almost_equal(float a, float b, float epsilon = 1e-6) {
    return (a == 0 && b == 0) || std::fabs(a - b) / b < epsilon;
}

std::string module = "benchmark_ntt_reduce";
constexpr size_t warmup_num = 10;
constexpr size_t run_num = 3000;
constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

#define NTT_REDUCEN_PACKN(mode)                                                \
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);               \
    NttTest::init_tensor(ta, -10.f, 10.f);                                     \
    auto taP =                                                                 \
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>); \
    ntt::vectorize(ta, taP.view(), ntt::fixed_shape_v<1>);                          \
                                                                               \
    auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<M, 1>);               \
                                                                               \
    for (size_t i = 0; i < warmup_num; i++) {                                  \
        ntt::reduce_##mode(taP, tb, ntt::fixed_shape_v<1>,                     \
                           ntt::fixed_shape_v<1>);                             \
    }                                                                          \
                                                                               \
    auto t1 = NttTest::get_cpu_cycle();                                        \
    for (size_t i = 0; i < run_num; i++) {                                     \
        ntt::reduce_##mode(taP, tb, ntt::fixed_shape_v<1>,                     \
                           ntt::fixed_shape_v<1>);                             \
        asm volatile("" ::"g"(tb));                                            \
    }                                                                          \
    auto t2 = NttTest::get_cpu_cycle();                                        \
                                                                               \
    std::ostringstream oss;                                                    \
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"      \
        << vectorize_mode << " took " << std::setprecision(0) << std::fixed         \
        << static_cast<float>(t2 - t1) / run_num << " cycles";                 \
    return oss.str();

#define NTT_REDUCEM_PACKM(mode)                                                \
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);               \
    NttTest::init_tensor(ta, -10.f, 10.f);                                     \
    auto taP =                                                                 \
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>); \
    ntt::vectorize(ta, taP.view(), ntt::fixed_shape_v<0>);                          \
                                                                               \
    auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<1, N>);               \
                                                                               \
    for (size_t i = 0; i < warmup_num; i++) {                                  \
        ntt::reduce_##mode(taP, tb, ntt::fixed_shape_v<0>,                     \
                           ntt::fixed_shape_v<0>);                             \
    }                                                                          \
                                                                               \
    auto t1 = NttTest::get_cpu_cycle();                                        \
    for (size_t i = 0; i < run_num; i++) {                                     \
        ntt::reduce_##mode(taP, tb, ntt::fixed_shape_v<0>,                     \
                           ntt::fixed_shape_v<0>);                             \
        asm volatile("" ::"g"(tb));                                            \
    }                                                                          \
    auto t2 = NttTest::get_cpu_cycle();                                        \
                                                                               \
    std::ostringstream oss;                                                    \
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"      \
        << vectorize_mode << " took " << std::setprecision(0) << std::fixed         \
        << static_cast<float>(t2 - t1) / run_num << " cycles";                 \
    return oss.str();

#define NTT_REDUCEMN_PACKN(mode)                                               \
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);               \
    NttTest::init_tensor(ta, -10.f, 10.f);                                     \
    auto taP =                                                                 \
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>); \
    ntt::vectorize(ta, taP.view(), ntt::fixed_shape_v<1>);                          \
                                                                               \
    auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1>);               \
                                                                               \
    for (size_t i = 0; i < warmup_num; i++) {                                  \
        ntt::reduce_##mode(taP, tb, ntt::fixed_shape_v<0, 1>,                  \
                           ntt::fixed_shape_v<1>);                             \
    }                                                                          \
                                                                               \
    auto t1 = NttTest::get_cpu_cycle();                                        \
    for (size_t i = 0; i < run_num; i++) {                                     \
        ntt::reduce_##mode(taP, tb, ntt::fixed_shape_v<0, 1>,                  \
                           ntt::fixed_shape_v<1>);                             \
        asm volatile("" ::"g"(tb));                                            \
    }                                                                          \
    auto t2 = NttTest::get_cpu_cycle();                                        \
                                                                               \
    std::ostringstream oss;                                                    \
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"      \
        << vectorize_mode << " took " << std::setprecision(0) << std::fixed         \
        << static_cast<float>(t2 - t1) / run_num << " cycles";                 \
    return oss.str();

#define NTT_REDUCEMN_PACKM(mode)                                               \
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);               \
    NttTest::init_tensor(ta, -10.f, 10.f);                                     \
    auto taP =                                                                 \
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>); \
    ntt::vectorize(ta, taP.view(), ntt::fixed_shape_v<0>);                          \
                                                                               \
    auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1>);               \
                                                                               \
    for (size_t i = 0; i < warmup_num; i++) {                                  \
        ntt::reduce_##mode(taP, tb, ntt::fixed_shape_v<0, 1>,                  \
                           ntt::fixed_shape_v<0>);                             \
    }                                                                          \
                                                                               \
    auto t1 = NttTest::get_cpu_cycle();                                        \
    for (size_t i = 0; i < run_num; i++) {                                     \
        ntt::reduce_##mode(taP, tb, ntt::fixed_shape_v<0, 1>,                  \
                           ntt::fixed_shape_v<0>);                             \
        asm volatile("" ::"g"(tb));                                            \
    }                                                                          \
    auto t2 = NttTest::get_cpu_cycle();                                        \
                                                                               \
    std::ostringstream oss;                                                    \
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"      \
        << vectorize_mode << " took " << std::setprecision(0) << std::fixed         \
        << static_cast<float>(t2 - t1) / run_num << " cycles";                 \
    return oss.str();

// 1,Add_reduceN_VectorizeN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceN_vectorizeN() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceN";
    std::string vectorize_mode = "VectorizeN";

    NTT_REDUCEN_PACKN(sum)
}

// 3,Add_reduceM_VectorizeM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceM_vectorizeM() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceM";
    std::string vectorize_mode = "VectorizeM";

    NTT_REDUCEM_PACKM(sum)
}

// 5,Add_reduceMN_VectorizeN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceMN_vectorizeN() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceMN";
    std::string vectorize_mode = "VectorizeN";

    NTT_REDUCEMN_PACKN(sum)
}

// 6,Add_reduceMN_VectorizeM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceMN_vectorizeM() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceMN";
    std::string vectorize_mode = "VectorizeM";

    NTT_REDUCEMN_PACKM(sum)
}

// 8,Max_reduceN_VectorizeN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceN_vectorizeN() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceN";
    std::string vectorize_mode = "VectorizeN";

    NTT_REDUCEN_PACKN(max)
}

// 10,Max_reduceM_VectorizeM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceM_vectorizeM() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceM";
    std::string vectorize_mode = "VectorizeM";

    NTT_REDUCEM_PACKM(max)
}

// 12,Max_reduceMN_VectorizeN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceMN_vectorizeN() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceMN";
    std::string vectorize_mode = "VectorizeN";

    NTT_REDUCEMN_PACKN(max)
}

// 13,Max_reduceMN_VectorizeM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceMN_vectorizeM() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceMN";
    std::string vectorize_mode = "VectorizeM";

    NTT_REDUCEMN_PACKM(max)
}

// 15,Min_reduceN_VectorizeN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceN_vectorizeN() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceN";
    std::string vectorize_mode = "VectorizeN";

    NTT_REDUCEN_PACKN(min)
}

// 17,Min_reduceM_VectorizeM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceM_vectorizeM() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceM";
    std::string vectorize_mode = "VectorizeM";

    NTT_REDUCEM_PACKM(min)
}

// 19,Min_reduceMN_VectorizeN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceMN_vectorizeN() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceMN";
    std::string vectorize_mode = "VectorizeN";

    NTT_REDUCEMN_PACKN(min)
}

// 20,Min_reduceMN_VectorizeM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceMN_vectorizeM() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceMN";
    std::string vectorize_mode = "VectorizeM";

    NTT_REDUCEMN_PACKM(min)
}

// 22,Mean_reduceN_VectorizeN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceN_vectorizeN() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceN";
    std::string vectorize_mode = "VectorizeN";

    NTT_REDUCEN_PACKN(mean)
}

// 24,Mean_reduceM_VectorizeM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceM_vectorizeM() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceM";
    std::string vectorize_mode = "VectorizeM";

    NTT_REDUCEM_PACKM(mean)
}

// 26,Mean_reduceMN_VectorizeN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceMN_vectorizeN() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceMN";
    std::string vectorize_mode = "VectorizeN";

    NTT_REDUCEMN_PACKN(mean)
}

// 27,Mean_reduceMN_VectorizeM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceMN_vectorizeM() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceMN";
    std::string vectorize_mode = "VectorizeM";

    NTT_REDUCEMN_PACKM(mean)
}

#define BENCHMARK_NTT_REDUCE(OP, REDUCE_AXIS, PACK_MODE, M, N)                 \
    benchmark_ntt_reduce_##OP##_reduce##REDUCE_AXIS##_##PACK_MODE<M, N>();     \
    benchmark_ntt_reduce_##OP##_reduce##REDUCE_AXIS##_##PACK_MODE<M, N>();     \
    benchmark_ntt_reduce_##OP##_reduce##REDUCE_AXIS##_##PACK_MODE<M, N>();     \
    result =                                                                   \
        benchmark_ntt_reduce_##OP##_reduce##REDUCE_AXIS##_##PACK_MODE<M, N>(); \
    std::cout << result << std::endl;

int main() {

    std::string result;

    BENCHMARK_NTT_REDUCE(add, MN, vectorizeM, 2048, 2)
    BENCHMARK_NTT_REDUCE(max, MN, vectorizeM, 2048, 2)
    BENCHMARK_NTT_REDUCE(min, MN, vectorizeM, 2048, 2)
    BENCHMARK_NTT_REDUCE(mean, MN, vectorizeM, 2048, 2)

    BENCHMARK_NTT_REDUCE(add, MN, vectorizeN, 2, 2048)
    BENCHMARK_NTT_REDUCE(max, MN, vectorizeN, 2, 2048)
    BENCHMARK_NTT_REDUCE(min, MN, vectorizeN, 2, 2048)
    BENCHMARK_NTT_REDUCE(mean, MN, vectorizeN, 2, 2048)

    BENCHMARK_NTT_REDUCE(add, M, vectorizeM, 2048, 2)
    BENCHMARK_NTT_REDUCE(max, M, vectorizeM, 2048, 2)
    BENCHMARK_NTT_REDUCE(min, M, vectorizeM, 2048, 2)
    BENCHMARK_NTT_REDUCE(mean, M, vectorizeM, 2048, 2)

    BENCHMARK_NTT_REDUCE(add, N, vectorizeN, 2, 2048)
    BENCHMARK_NTT_REDUCE(max, N, vectorizeN, 2, 2048)
    BENCHMARK_NTT_REDUCE(min, N, vectorizeN, 2, 2048)
    BENCHMARK_NTT_REDUCE(mean, N, vectorizeN, 2, 2048)

    return 0;
}