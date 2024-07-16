#include "ntt_test.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

bool almost_equal(float a, float b, float epsilon = 1e-6) {
    return (a == 0 && b == 0) || std::fabs(a - b) / b < epsilon;
}

int main() {
    std::string module = "benchmark_ntt_reduce";
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
#if __riscv
    constexpr size_t M = 32;
    constexpr size_t N = 32;
#else
    constexpr size_t M = 256;
    constexpr size_t N = 256;
#endif

    // 0,Add_reduceN_NoPack
    {
        std::string reduce_mode = "Add";
        std::string reduce_direction = "reduceN";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::add>(ta, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::add>(ta, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 1,Add_reduceN_PackN
    {
        std::string reduce_mode = "Add";
        std::string reduce_direction = "reduceN";
        std::string pack_mode = "PackN";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
        ntt::pack<1>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 2,Add_reduceM_NoPack
    {
        std::string reduce_mode = "Add";
        std::string reduce_direction = "reduceM";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<1, N>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::add>(ta, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::add>(ta, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 3,Add_reduceM_PackM
    {
        std::string reduce_mode = "Add";
        std::string reduce_direction = "reduceM";
        std::string pack_mode = "PackM";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
        ntt::pack<0>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, N>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 4,Add_reduceMN_NoPack
    {
        std::string reduce_mode = "Add";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::add>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::add>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 5,Add_reduceMN_PackN
    {
        std::string reduce_mode = "Add";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "PackN";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
        ntt::pack<1>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 6,Add_reduceMN_PackM
    {
        std::string reduce_mode = "Add";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "PackM";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
        ntt::pack<0>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::add>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 7,Max_reduceN_NoPack
    {
        std::string reduce_mode = "Max";
        std::string reduce_direction = "reduceN";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::max>(ta, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::max>(ta, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 8,Max_reduceN_PackN
    {
        std::string reduce_mode = "Max";
        std::string reduce_direction = "reduceN";
        std::string pack_mode = "PackN";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
        ntt::pack<1>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 9,Max_reduceM_NoPack
    {
        std::string reduce_mode = "Max";
        std::string reduce_direction = "reduceM";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<1, N>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::max>(ta, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::max>(ta, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 10,Max_reduceM_PackM
    {
        std::string reduce_mode = "Max";
        std::string reduce_direction = "reduceM";
        std::string pack_mode = "PackM";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
        ntt::pack<0>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, N>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 11,Max_reduceMN_NoPack
    {
        std::string reduce_mode = "Max";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::max>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::max>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 12,Max_reduceMN_PackN
    {
        std::string reduce_mode = "Max";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "PackN";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
        ntt::pack<1>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 13,Max_reduceMN_PackM
    {
        std::string reduce_mode = "Max";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "PackM";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
        ntt::pack<0>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::max>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 14,Min_reduceN_NoPack
    {
        std::string reduce_mode = "Min";
        std::string reduce_direction = "reduceN";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::min>(ta, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::min>(ta, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 15,Min_reduceN_PackN
    {
        std::string reduce_mode = "Min";
        std::string reduce_direction = "reduceN";
        std::string pack_mode = "PackN";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
        ntt::pack<1>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 16,Min_reduceM_NoPack
    {
        std::string reduce_mode = "Min";
        std::string reduce_direction = "reduceM";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<1, N>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::min>(ta, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::min>(ta, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 17,Min_reduceM_PackM
    {
        std::string reduce_mode = "Min";
        std::string reduce_direction = "reduceM";
        std::string pack_mode = "PackM";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
        ntt::pack<0>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, N>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 18,Min_reduceMN_NoPack
    {
        std::string reduce_mode = "Min";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::min>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::min>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 19,Min_reduceMN_PackN
    {
        std::string reduce_mode = "Min";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "PackN";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
        ntt::pack<1>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<1>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 20,Min_reduceMN_PackM
    {
        std::string reduce_mode = "Min";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "PackM";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
        ntt::pack<0>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::min>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                       ntt::fixed_shape<0>{},
                                       ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 21,Mean_reduceN_NoPack
    {
        std::string reduce_mode = "Mean";
        std::string reduce_direction = "reduceN";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::mean>(ta, tb[i], ntt::fixed_shape<1>{},
                                        ntt::fixed_shape<>{},
                                        ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::mean>(ta, tb[i], ntt::fixed_shape<1>{},
                                        ntt::fixed_shape<>{},
                                        ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 22,Mean_reduceN_PackN
    {
        std::string reduce_mode = "Mean";
        std::string reduce_direction = "reduceN";
        std::string pack_mode = "PackN";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
        ntt::pack<1>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<M, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<1>{},
                                        ntt::fixed_shape<1>{},
                                        ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<1>{},
                                        ntt::fixed_shape<1>{},
                                        ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 23,Mean_reduceM_NoPack
    {
        std::string reduce_mode = "Mean";
        std::string reduce_direction = "reduceM";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<1, N>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::mean>(ta, tb[i], ntt::fixed_shape<0>{},
                                        ntt::fixed_shape<>{},
                                        ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::mean>(ta, tb[i], ntt::fixed_shape<0>{},
                                        ntt::fixed_shape<>{},
                                        ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 24,Mean_reduceM_PackM
    {
        std::string reduce_mode = "Mean";
        std::string reduce_direction = "reduceM";
        std::string pack_mode = "PackM";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
        ntt::pack<0>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, N>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<0>{},
                                        ntt::fixed_shape<0>{},
                                        ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<0>{},
                                        ntt::fixed_shape<0>{},
                                        ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 25,Mean_reduceMN_NoPack
    {
        std::string reduce_mode = "Mean";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "NoPack";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::mean>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                        ntt::fixed_shape<>{},
                                        ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::mean>(ta, tb[i], ntt::fixed_shape<0, 1>{},
                                        ntt::fixed_shape<>{},
                                        ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 26,Mean_reduceMN_PackN
    {
        std::string reduce_mode = "Mean";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "PackN";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>> taP;
        ntt::pack<1>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                        ntt::fixed_shape<1>{},
                                        ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                        ntt::fixed_shape<1>{},
                                        ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    // 27,Mean_reduceMN_PackM
    {
        std::string reduce_mode = "Mean";
        std::string reduce_direction = "reduceMN";
        std::string pack_mode = "PackM";

        ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
        NttTest::init_tensor(ta, -10.f, 10.f);
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>> taP;
        ntt::pack<0>(ta, taP.view());

        ntt::tensor<float, ntt::fixed_shape<1, 1>> tb[run_num];

        for (size_t i = 0; i < warmup_num; i++) {
            ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                        ntt::fixed_shape<0>{},
                                        ntt::fixed_shape<>{});
        }

        auto t1 = NttTest::get_cpu_cycle();
        for (size_t i = 0; i < run_num; i++) {
            ntt::reduce<ntt::ops::mean>(taP, tb[i], ntt::fixed_shape<0, 1>{},
                                        ntt::fixed_shape<0>{},
                                        ntt::fixed_shape<>{});
        }
        auto t2 = NttTest::get_cpu_cycle();

        std::cout << module << "_"
                  << reduce_mode + "_" + reduce_direction + "_" + pack_mode
                  << " took " << std::setprecision(0) << std::fixed
                  << static_cast<float>(t2 - t1) / run_num << " cycles"
                  << std::endl;
    }

    return 0;
}