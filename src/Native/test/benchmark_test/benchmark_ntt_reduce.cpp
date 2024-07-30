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
#if __riscv
constexpr size_t M = 256;
constexpr size_t N = 256;
#else
constexpr size_t M = 256;
constexpr size_t N = 256;
#endif

// 0,Add_reduceN_NoPack
std::string benchmark_ntt_reduce_Add_reduceN_noPack() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::add>(ta, tb[i], ntt::fixed_shape<1>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::add>(ta, tb[warmup_num + i],
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 1,Add_reduceN_PackN
std::string benchmark_ntt_reduce_Add_reduceN_packN() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<1>{},
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::add>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<1>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 2,Add_reduceM_NoPack
std::string benchmark_ntt_reduce_Add_reduceM_noPack() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::add>(ta, tb[i], ntt::fixed_shape<0>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::add>(ta, tb[warmup_num + i],
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 3,Add_reduceM_PackM
std::string benchmark_ntt_reduce_Add_reduceM_packM() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<0>{},
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::add>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 4,Add_reduceMN_NoPack
std::string benchmark_ntt_reduce_Add_reduceMN_noPack() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::add>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::add>(ta, tb[warmup_num + i],
                                   ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 5,Add_reduceMN_PackN
std::string benchmark_ntt_reduce_Add_reduceMN_packN() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::add>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 6,Add_reduceMN_PackM
std::string benchmark_ntt_reduce_Add_reduceMN_packM() {
    std::string reduce_mode = "Add";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::add>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 7,Max_reduceN_NoPack
std::string benchmark_ntt_reduce_Max_reduceN_noPack() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::max>(ta, tb[i], ntt::fixed_shape<1>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::max>(ta, tb[warmup_num + i],
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 8,Max_reduceN_PackN
std::string benchmark_ntt_reduce_Max_reduceN_packN() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<1>{},
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::max>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<1>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 9,Max_reduceM_NoPack
std::string benchmark_ntt_reduce_Max_reduceM_noPack() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::max>(ta, tb[i], ntt::fixed_shape<0>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::max>(ta, tb[warmup_num + i],
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 10,Max_reduceM_PackM
std::string benchmark_ntt_reduce_Max_reduceM_packM() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<0>{},
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::max>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 11,Max_reduceMN_NoPack
std::string benchmark_ntt_reduce_Max_reduceMN_noPack() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::max>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::max>(ta, tb[warmup_num + i],
                                   ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 12,Max_reduceMN_PackN
std::string benchmark_ntt_reduce_Max_reduceMN_packN() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::max>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 13,Max_reduceMN_PackM
std::string benchmark_ntt_reduce_Max_reduceMN_packM() {
    std::string reduce_mode = "Max";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::max>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 14,Min_reduceN_NoPack
std::string benchmark_ntt_reduce_Min_reduceN_noPack() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::min>(ta, tb[i], ntt::fixed_shape<1>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::min>(ta, tb[warmup_num + i],
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 15,Min_reduceN_PackN
std::string benchmark_ntt_reduce_Min_reduceN_packN() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<1>{},
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::min>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<1>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 16,Min_reduceM_NoPack
std::string benchmark_ntt_reduce_Min_reduceM_noPack() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::min>(ta, tb[i], ntt::fixed_shape<0>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::min>(ta, tb[warmup_num + i],
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 17,Min_reduceM_PackM
std::string benchmark_ntt_reduce_Min_reduceM_packM() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<0>{},
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::min>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                                   ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 18,Min_reduceMN_NoPack

std::string benchmark_ntt_reduce_Min_reduceMN_noPack() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::min>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::min>(ta, tb[warmup_num + i],
                                   ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 19,Min_reduceMN_PackN
std::string benchmark_ntt_reduce_Min_reduceMN_packN() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::min>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 20,Min_reduceMN_PackM
std::string benchmark_ntt_reduce_Min_reduceMN_packM() {
    std::string reduce_mode = "Min";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::min>(taP, tb[warmup_num + i],
                                   ntt::fixed_shape<0, 1>{},
                                   ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 21,Mean_reduceN_NoPack
std::string benchmark_ntt_reduce_Mean_reduceN_noPack() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::mean>(ta, tb[i], ntt::fixed_shape<1>{},
                                    ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::mean>(ta, tb[warmup_num + i],
                                    ntt::fixed_shape<1>{}, ntt::fixed_shape<>{},
                                    ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 22,Mean_reduceN_PackN
std::string benchmark_ntt_reduce_Mean_reduceN_packN() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<1>{},
                                    ntt::fixed_shape<1>{},
                                    ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::mean>(
            taP, tb[warmup_num + i], ntt::fixed_shape<1>{},
            ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 23,Mean_reduceM_NoPack
std::string benchmark_ntt_reduce_Mean_reduceM_noPack() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::mean>(ta, tb[i], ntt::fixed_shape<0>{},
                                    ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::mean>(ta, tb[warmup_num + i],
                                    ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                                    ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 24,Mean_reduceM_PackM
std::string benchmark_ntt_reduce_Mean_reduceM_packM() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceM";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, N>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<0>{},
                                    ntt::fixed_shape<0>{},
                                    ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::mean>(
            taP, tb[warmup_num + i], ntt::fixed_shape<0>{},
            ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 25,Mean_reduceMN_NoPack
std::string benchmark_ntt_reduce_Mean_reduceMN_noPack() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "NoPack";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::mean>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                    ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::mean>(ta, tb[warmup_num + i],
                                    ntt::fixed_shape<0, 1>{},
                                    ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 26,Mean_reduceMN_PackN
std::string benchmark_ntt_reduce_Mean_reduceMN_packN() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackN";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
    ntt::pack<1>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                    ntt::fixed_shape<1>{},
                                    ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::mean>(
            taP, tb[warmup_num + i], ntt::fixed_shape<0, 1>{},
            ntt::fixed_shape<1>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

// 27,Mean_reduceMN_PackM
std::string benchmark_ntt_reduce_Mean_reduceMN_packM() {
    std::string reduce_mode = "Mean";
    std::string reduce_direction = "reduceMN";
    std::string pack_mode = "PackM";

    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    NttTest::init_tensor(ta, -10.f, 10.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
    ntt::pack<0>(ta, taP.view());

    ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[warmup_num + run_num];

    for (size_t i = 0; i < warmup_num; i++) {
        ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                    ntt::fixed_shape<0>{},
                                    ntt::fixed_shape<>{});
    }

    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_num; i++) {
        ntt::reduce<ntt::ops::mean>(
            taP, tb[warmup_num + i], ntt::fixed_shape<0, 1>{},
            ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::ostringstream oss;
    oss << module << "_" << reduce_mode << "_" << reduce_direction << "_"
        << pack_mode << " took " << std::setprecision(0) << std::fixed
        << static_cast<float>(t2 - t1) / run_num << " cycles";
    return oss.str();
}

#define BENCHMARK_NTT_REDUCE(OP, REDUCE_AXIS, PACK_MODE)                       \
    result = benchmark_ntt_reduce_##OP##_reduce##REDUCE_AXIS##_##PACK_MODE();  \
    std::cout << result << std::endl;

int main() {

    std::string result;

    BENCHMARK_NTT_REDUCE(Add, MN, noPack)
    BENCHMARK_NTT_REDUCE(Max, MN, noPack)
    BENCHMARK_NTT_REDUCE(Min, MN, noPack)
    BENCHMARK_NTT_REDUCE(Mean, MN, noPack)

    BENCHMARK_NTT_REDUCE(Add, MN, packM)
    BENCHMARK_NTT_REDUCE(Max, MN, packM)
    BENCHMARK_NTT_REDUCE(Min, MN, packM)
    BENCHMARK_NTT_REDUCE(Mean, MN, packM)

    BENCHMARK_NTT_REDUCE(Add, MN, packN)
    BENCHMARK_NTT_REDUCE(Max, MN, packN)
    BENCHMARK_NTT_REDUCE(Min, MN, packN)
    BENCHMARK_NTT_REDUCE(Mean, MN, packN)

    BENCHMARK_NTT_REDUCE(Add, M, noPack)
    BENCHMARK_NTT_REDUCE(Max, M, noPack)
    BENCHMARK_NTT_REDUCE(Min, M, noPack)
    BENCHMARK_NTT_REDUCE(Mean, M, noPack)

    BENCHMARK_NTT_REDUCE(Add, N, noPack)
    BENCHMARK_NTT_REDUCE(Max, N, noPack)
    BENCHMARK_NTT_REDUCE(Min, N, noPack)
    BENCHMARK_NTT_REDUCE(Mean, N, noPack)

    BENCHMARK_NTT_REDUCE(Add, M, packM)
    BENCHMARK_NTT_REDUCE(Max, M, packM)
    BENCHMARK_NTT_REDUCE(Min, M, packM)
    BENCHMARK_NTT_REDUCE(Mean, M, packM)

    BENCHMARK_NTT_REDUCE(Add, N, packN)
    BENCHMARK_NTT_REDUCE(Max, N, packN)
    BENCHMARK_NTT_REDUCE(Min, N, packN)
    BENCHMARK_NTT_REDUCE(Mean, N, packN)

    return 0;
}