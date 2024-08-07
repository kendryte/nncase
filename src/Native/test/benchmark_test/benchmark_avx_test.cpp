#include "ntt_test.h"
#include <iomanip>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

bool are_floats_equal(float a, float b, float epsilon = 1e-6) {
    return std::fabs(a - b) < epsilon;
}

template <typename T>
concept is_vector = T::rank() == 1;

template <typename T>
concept is_tensor = T::rank() > 1;

// 一维数组处理函数
template <is_vector T> void processNTT([[maybe_unused]] T &array) {
    printf("vector processor\n");
}

// 二维数组处理函数
template <is_tensor T> void processNTT([[maybe_unused]] T &array) {
    printf("matrix processor\n");
}

int main() {

#if 1
    {
        using typeNoPack = ntt::tensor<float, ntt::fixed_shape<8, 8>>;
        typeNoPack ti, to;
        std::iota(ti.elements().begin(), ti.elements().end(), -32.f);
        std::fill(to.elements().begin(), to.elements().end(), 0.f);

        ntt::clamp(ti, to, -30.f, 30.f);

        assert(to(0, 0) == -30.f);
        assert(to(0, 1) == -30.f);
        assert(to(0, 2) == -30.f);
        assert(to(0, 3) == -29.f);
        assert(to(0, 4) == -28.f);
        assert(to(7, 3) == 27.f);
        assert(to(7, 4) == 28.f);
        assert(to(7, 5) == 29.f);
        assert(to(7, 6) == 30.f);
        assert(to(7, 7) == 30.f);
    }

    {
        using typeNoPack = ntt::tensor<float, ntt::fixed_shape<8, 8>>;
        typeNoPack ti, to;
        std::iota(ti.elements().begin(), ti.elements().end(), -32.f);
        std::fill(to.elements().begin(), to.elements().end(), 0.f);

        using typePack =
            ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<8, 1>>;

        typePack pi, po;
        ntt::pack<1>(ti, pi);
        ntt::pack<1>(to, po);

        ntt::clamp(pi, po, -30.f, 30.f);

        assert(po(0, 0)(0) == -30.f);
        assert(po(0, 0)(1) == -30.f);
        assert(po(0, 0)(2) == -30.f);
        assert(po(0, 0)(3) == -29.f);
        assert(po(0, 0)(4) == -28.f);
        assert(po(7, 0)(3) == 27.f);
        assert(po(7, 0)(4) == 28.f);
        assert(po(7, 0)(5) == 29.f);
        assert(po(7, 0)(6) == 30.f);
        assert(po(7, 0)(7) == 30.f);
    }

    {
        using typeNoPack = ntt::tensor<float, ntt::fixed_shape<8, 8>>;
        typeNoPack ti, to;
        std::iota(ti.elements().begin(), ti.elements().end(), -32.f);
        std::fill(to.elements().begin(), to.elements().end(), 0.f);

        using typePack =
            ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 8>>;

        typePack pi, po;
        ntt::pack<0>(ti, pi);
        ntt::pack<0>(to, po);

        ntt::clamp(pi, po, -30.f, 30.f);

        assert(po(0, 0)(0) == -30.f);
        assert(po(0, 0)(1) == -24.f);
        assert(po(0, 0)(2) == -16.f);
        assert(po(0, 0)(3) == -8.f);
        assert(po(0, 0)(4) == -0.f);
        assert(po(0, 7)(3) == -1.f);
        assert(po(0, 7)(4) == 7.f);
        assert(po(0, 7)(5) == 15.f);
        assert(po(0, 7)(6) == 23.f);
        assert(po(0, 7)(7) == 30.f);
    }

#endif
    {
        using typeNoPack = ntt::tensor<float, ntt::fixed_shape<256, 256>>;
        typeNoPack ti, to;
        std::iota(ti.elements().begin(), ti.elements().end(), -32.f);
        std::fill(to.elements().begin(), to.elements().end(), 0.f);

        using typePack =
            ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<32, 256>>;

        typePack pi, po;
        ntt::pack<0>(ti, pi);
        ntt::pack<0>(to, po);

        auto t1 = NttTest::get_cpu_cycle();
        ntt::clamp(pi, po, -30.f, 30.f);
        auto t2 = NttTest::get_cpu_cycle();
        std::cout << __FUNCTION__ << " took " << std::setprecision(0)
                  << std::fixed << static_cast<float>(t2 - t1) << " cycles"
                  << std::endl;

        assert(po(0, 0)(0) == -30.f);
        assert(po(0, 0)(1) == -24.f);
        assert(po(0, 0)(2) == -16.f);
        assert(po(0, 0)(3) == -8.f);
        assert(po(0, 0)(4) == -0.f);
        assert(po(0, 7)(3) == -1.f);
        assert(po(0, 7)(4) == 7.f);
        assert(po(0, 7)(5) == 15.f);
        assert(po(0, 7)(6) == 23.f);
        assert(po(0, 7)(7) == 30.f);
    }

    return 0;
}