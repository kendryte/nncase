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

// 0,Add_reduceN_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceN_noPack() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<1>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<1>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 1,Add_reduceN_PackN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceN_packN() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<1>, ntt::fixed_shape<1>>(taP, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<1>, ntt::fixed_shape<1>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 2,Add_reduceM_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceM_noPack() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 3,Add_reduceM_PackM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceM_packM() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0>, ntt::fixed_shape<0>>(taP, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0>, ntt::fixed_shape<0>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 4,Add_reduceMN_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceMN_noPack() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0, 1>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0, 1>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 5,Add_reduceMN_PackN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceMN_packN() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0, 1>, ntt::fixed_shape<1>>(taP,
                                                                     tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0, 1>, ntt::fixed_shape<1>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 6,Add_reduceMN_PackM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_add_reduceMN_packM() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0, 1>, ntt::fixed_shape<0>>(taP,
                                                                     tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_sum<ntt::fixed_shape<0, 1>, ntt::fixed_shape<0>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 7,Max_reduceN_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceN_noPack() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<1>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<1>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 8,Max_reduceN_PackN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceN_packN() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<1>, ntt::fixed_shape<1>>(taP, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<1>, ntt::fixed_shape<1>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 9,Max_reduceM_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceM_noPack() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 10,Max_reduceM_PackM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceM_packM() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0>, ntt::fixed_shape<0>>(taP, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0>, ntt::fixed_shape<0>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 11,Max_reduceMN_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceMN_noPack() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0, 1>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0, 1>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 12,Max_reduceMN_PackN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceMN_packN() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0, 1>, ntt::fixed_shape<1>>(taP,
                                                                     tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0, 1>, ntt::fixed_shape<1>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 13,Max_reduceMN_PackM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_max_reduceMN_packM() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0, 1>, ntt::fixed_shape<0>>(taP,
                                                                     tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_max<ntt::fixed_shape<0, 1>, ntt::fixed_shape<0>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 14,Min_reduceN_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceN_noPack() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<1>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<1>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 15,Min_reduceN_PackN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceN_packN() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<1>, ntt::fixed_shape<1>>(taP, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<1>, ntt::fixed_shape<1>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 16,Min_reduceM_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceM_noPack() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 17,Min_reduceM_PackM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceM_packM() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0>, ntt::fixed_shape<0>>(taP, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0>, ntt::fixed_shape<0>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 18,Min_reduceMN_NoPack

template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceMN_noPack() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0, 1>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0, 1>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 19,Min_reduceMN_PackN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceMN_packN() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0, 1>, ntt::fixed_shape<1>>(taP,
                                                                     tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0, 1>, ntt::fixed_shape<1>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 20,Min_reduceMN_PackM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_min_reduceMN_packM() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0, 1>, ntt::fixed_shape<0>>(taP,
                                                                     tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_min<ntt::fixed_shape<0, 1>, ntt::fixed_shape<0>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 21,Mean_reduceN_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceN_noPack() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<1>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<1>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 22,Mean_reduceN_PackN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceN_packN() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<1>, ntt::fixed_shape<1>>(taP, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<1>, ntt::fixed_shape<1>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 23,Mean_reduceM_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceM_noPack() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 24,Mean_reduceM_PackM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceM_packM() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0>, ntt::fixed_shape<0>>(taP, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0>, ntt::fixed_shape<0>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 25,Mean_reduceMN_NoPack
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceMN_noPack() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0, 1>>(ta, tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0, 1>>(ta, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 26,Mean_reduceMN_PackN
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceMN_packN() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0, 1>, ntt::fixed_shape<1>>(taP,
                                                                      tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0, 1>, ntt::fixed_shape<1>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 27,Mean_reduceMN_PackM
template <size_t M, size_t N>
std::string benchmark_ntt_reduce_mean_reduceMN_packM() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0, 1>, ntt::fixed_shape<0>>(taP,
                                                                      tb[i]);
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce_mean<ntt::fixed_shape<0, 1>, ntt::fixed_shape<0>>(
            taP, tb[warmup_num + i]);
    }
    auto t2 = NttTest::get_cpu_cycle();
    asm volatile("" ::"g"(tb));

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
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

    BENCHMARK_NTT_REDUCE(add, MN, noPack, 64, 64)
    BENCHMARK_NTT_REDUCE(max, MN, noPack, 64, 64)
    BENCHMARK_NTT_REDUCE(min, MN, noPack, 64, 64)
    BENCHMARK_NTT_REDUCE(mean, MN, noPack, 64, 64)

    BENCHMARK_NTT_REDUCE(add, MN, packM, 2048, 2)
    BENCHMARK_NTT_REDUCE(max, MN, packM, 2048, 2)
    BENCHMARK_NTT_REDUCE(min, MN, packM, 2048, 2)
    BENCHMARK_NTT_REDUCE(mean, MN, packM, 2048, 2)

    BENCHMARK_NTT_REDUCE(add, MN, packN, 2, 2048)
    BENCHMARK_NTT_REDUCE(max, MN, packN, 2, 2048)
    BENCHMARK_NTT_REDUCE(min, MN, packN, 2, 2048)
    BENCHMARK_NTT_REDUCE(mean, MN, packN, 2, 2048)

    BENCHMARK_NTT_REDUCE(add, M, noPack, 2048, 2)
    BENCHMARK_NTT_REDUCE(max, M, noPack, 2048, 2)
    BENCHMARK_NTT_REDUCE(min, M, noPack, 2048, 2)
    BENCHMARK_NTT_REDUCE(mean, M, noPack, 2048, 2)

    BENCHMARK_NTT_REDUCE(add, N, noPack, 2, 2048)
    BENCHMARK_NTT_REDUCE(max, N, noPack, 2, 2048)
    BENCHMARK_NTT_REDUCE(min, N, noPack, 2, 2048)
    BENCHMARK_NTT_REDUCE(mean, N, noPack, 2, 2048)

    BENCHMARK_NTT_REDUCE(add, M, packM, 2048, 2)
    BENCHMARK_NTT_REDUCE(max, M, packM, 2048, 2)
    BENCHMARK_NTT_REDUCE(min, M, packM, 2048, 2)
    BENCHMARK_NTT_REDUCE(mean, M, packM, 2048, 2)

    BENCHMARK_NTT_REDUCE(add, N, packN, 2, 2048)
    BENCHMARK_NTT_REDUCE(max, N, packN, 2, 2048)
    BENCHMARK_NTT_REDUCE(min, N, packN, 2, 2048)
    BENCHMARK_NTT_REDUCE(mean, N, packN, 2, 2048)

    return 0;
}