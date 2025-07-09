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
#include "../../vector_ops.h"
#include "arch_types.h"

namespace nncase::ntt::vector_ops {
#define RVV_LOAD_SCALAR_FLOAT32(vl, lmul)                                      \
    template <> struct vload_scalar<ntt::vector<float, vl>> {                  \
        ntt::vector<float, vl> operator()(float f) const noexcept {            \
            return __riscv_vfmv_v_f_f32m##lmul(f, vl);                         \
        }                                                                      \
    };

RVV_LOAD_SCALAR_FLOAT32(NTT_VL(sizeof(float) * 8, *, 1), 1)
RVV_LOAD_SCALAR_FLOAT32(NTT_VL(sizeof(float) * 8, *, 2), 2)
RVV_LOAD_SCALAR_FLOAT32(NTT_VL(sizeof(float) * 8, *, 4), 4)
RVV_LOAD_SCALAR_FLOAT32(NTT_VL(sizeof(float) * 8, *, 8), 8)

template <bool AccC>
struct vmma<AccC, false, ntt::vector<float, 1, 4>, ntt::vector<float, 4, 4>,
            ntt::vector<float, 1, 4>> {
    ntt::vector<float, 1, 4>
    operator()(const ntt::vector<float, 1, 4> &lhs,
               const ntt::vector<float, 4, 4> &rhs,
               const ntt::vector<float, 1, 4> &v3) const noexcept {
        auto output = v3;
        auto t0 = AccC ? ntt::mul_add(lhs(0, 0), rhs(0), output(0))
                       : ntt::mul(lhs(0, 0), rhs(0));
        auto t1 = ntt::mul(lhs(0, 1), rhs(1));
        t0 = ntt::mul_add(lhs(0, 2), rhs(2), t0);
        t1 = ntt::mul_add(lhs(0, 3), rhs(3), t1);
        output(0) = ntt::add(t0, t1);
        return output;
    }
};

template <bool AccC>
struct vmma<AccC, false, ntt::vector<float, 1, 32>, ntt::vector<float, 32, 32>,
            ntt::vector<float, 1, 32>> {
    ntt::vector<float, 1, 32>
    operator()(const ntt::vector<float, 1, 32> &lhs,
               const ntt::vector<float, 32, 32> &rhs,
               const ntt::vector<float, 1, 32> &v3) const noexcept {
        auto output = v3;

        auto t0 = AccC ? ntt::mul_add(lhs(0, 0), rhs(0), output(0))
                       : ntt::mul(lhs(0, 0), rhs(0));
        auto t1 = ntt::mul(lhs(0, 1), rhs(1));
        t0 = ntt::mul_add(lhs(0, 2), rhs(2), t0);
        t1 = ntt::mul_add(lhs(0, 3), rhs(3), t1);

        t0 = ntt::mul_add(lhs(0, 4), rhs(4), t0);
        t1 = ntt::mul_add(lhs(0, 5), rhs(5), t1);
        t0 = ntt::mul_add(lhs(0, 6), rhs(6), t0);
        t1 = ntt::mul_add(lhs(0, 7), rhs(7), t1);

        t0 = ntt::mul_add(lhs(0, 8), rhs(8), t0);
        t1 = ntt::mul_add(lhs(0, 9), rhs(9), t1);
        t0 = ntt::mul_add(lhs(0, 10), rhs(10), t0);
        t1 = ntt::mul_add(lhs(0, 11), rhs(11), t1);

        t0 = ntt::mul_add(lhs(0, 12), rhs(12), t0);
        t1 = ntt::mul_add(lhs(0, 13), rhs(13), t1);
        t0 = ntt::mul_add(lhs(0, 14), rhs(14), t0);
        t1 = ntt::mul_add(lhs(0, 15), rhs(15), t1);

        t0 = ntt::mul_add(lhs(0, 16), rhs(16), t0);
        t1 = ntt::mul_add(lhs(0, 17), rhs(17), t1);
        t0 = ntt::mul_add(lhs(0, 18), rhs(18), t0);
        t1 = ntt::mul_add(lhs(0, 19), rhs(19), t1);

        t0 = ntt::mul_add(lhs(0, 20), rhs(20), t0);
        t1 = ntt::mul_add(lhs(0, 21), rhs(21), t1);
        t0 = ntt::mul_add(lhs(0, 22), rhs(22), t0);
        t1 = ntt::mul_add(lhs(0, 23), rhs(23), t1);

        t0 = ntt::mul_add(lhs(0, 24), rhs(24), t0);
        t1 = ntt::mul_add(lhs(0, 25), rhs(25), t1);
        t0 = ntt::mul_add(lhs(0, 26), rhs(26), t0);
        t1 = ntt::mul_add(lhs(0, 27), rhs(27), t1);

        t0 = ntt::mul_add(lhs(0, 28), rhs(28), t0);
        t1 = ntt::mul_add(lhs(0, 29), rhs(29), t1);
        t0 = ntt::mul_add(lhs(0, 30), rhs(30), t0);
        t1 = ntt::mul_add(lhs(0, 31), rhs(31), t1);

        output(0) = ntt::add(t0, t1);
        return output;
    }
};

template <bool AccC>
struct vmma<AccC, false, ntt::vector<float, 4, 4>, ntt::vector<float, 4, 4>,
            ntt::vector<float, 4, 4>> {
    ntt::vector<float, 4, 4>
    operator()(const ntt::vector<float, 4, 4> &lhs,
               const ntt::vector<float, 4, 4> &rhs,
               const ntt::vector<float, 4, 4> &v3) const noexcept {
        auto output = v3;
        ntt::vector<float, 1, 4> lhs_2d[4]{
            {{lhs(0)}},
            {{lhs(1)}},
            {{lhs(2)}},
            {{lhs(3)}},
        };
        ntt::vector<float, 1, 4> output_2d[4]{
            {{v3(0)}},
            {{v3(1)}},
            {{v3(2)}},
            {{v3(3)}},
        };

        output_2d[0] = ntt::vmma<AccC>(lhs_2d[0], rhs, output_2d[0]);
        output_2d[1] = ntt::vmma<AccC>(lhs_2d[1], rhs, output_2d[1]);
        output_2d[2] = ntt::vmma<AccC>(lhs_2d[2], rhs, output_2d[2]);
        output_2d[3] = ntt::vmma<AccC>(lhs_2d[3], rhs, output_2d[3]);

        output(0) = output_2d[0](0);
        output(1) = output_2d[1](0);
        output(2) = output_2d[2](0);
        output(3) = output_2d[3](0);

        return output;
    }
};

template <bool AccC>
struct vmma<AccC, false, ntt::vector<float, 32, 32>, ntt::vector<float, 32, 32>,
            ntt::vector<float, 32, 32>> {
    ntt::vector<float, 32, 32>
    operator()(const ntt::vector<float, 32, 32> &lhs,
               const ntt::vector<float, 32, 32> &rhs,
               const ntt::vector<float, 32, 32> &v3) const noexcept {
        auto output = v3;
        ntt::vector<float, 1, 32> lhs_2d[]{
            {{lhs(0)}},  {{lhs(1)}},  {{lhs(2)}},  {{lhs(3)}},  {{lhs(4)}},
            {{lhs(5)}},  {{lhs(6)}},  {{lhs(7)}},  {{lhs(8)}},  {{lhs(9)}},
            {{lhs(10)}}, {{lhs(11)}}, {{lhs(12)}}, {{lhs(13)}}, {{lhs(14)}},
            {{lhs(15)}}, {{lhs(16)}}, {{lhs(17)}}, {{lhs(18)}}, {{lhs(19)}},
            {{lhs(20)}}, {{lhs(21)}}, {{lhs(22)}}, {{lhs(23)}}, {{lhs(24)}},
            {{lhs(25)}}, {{lhs(26)}}, {{lhs(27)}}, {{lhs(28)}}, {{lhs(29)}},
            {{lhs(30)}}, {{lhs(31)}}};

        ntt::vector<float, 1, 32> output_2d[]{
            {{v3(0)}},  {{v3(1)}},  {{v3(2)}},  {{v3(3)}},  {{v3(4)}},
            {{v3(5)}},  {{v3(6)}},  {{v3(7)}},  {{v3(8)}},  {{v3(9)}},
            {{v3(10)}}, {{v3(11)}}, {{v3(12)}}, {{v3(13)}}, {{v3(14)}},
            {{v3(15)}}, {{v3(16)}}, {{v3(17)}}, {{v3(18)}}, {{v3(19)}},
            {{v3(20)}}, {{v3(21)}}, {{v3(22)}}, {{v3(23)}}, {{v3(24)}},
            {{v3(25)}}, {{v3(26)}}, {{v3(27)}}, {{v3(28)}}, {{v3(29)}},
            {{v3(30)}}, {{v3(31)}}};

        output_2d[0] = ntt::vmma<AccC>(lhs_2d[0], rhs, output_2d[0]);
        output_2d[1] = ntt::vmma<AccC>(lhs_2d[1], rhs, output_2d[1]);
        output_2d[2] = ntt::vmma<AccC>(lhs_2d[2], rhs, output_2d[2]);
        output_2d[3] = ntt::vmma<AccC>(lhs_2d[3], rhs, output_2d[3]);
        output_2d[4] = ntt::vmma<AccC>(lhs_2d[4], rhs, output_2d[4]);
        output_2d[5] = ntt::vmma<AccC>(lhs_2d[5], rhs, output_2d[5]);
        output_2d[6] = ntt::vmma<AccC>(lhs_2d[6], rhs, output_2d[6]);
        output_2d[7] = ntt::vmma<AccC>(lhs_2d[7], rhs, output_2d[7]);

        output_2d[8] = ntt::vmma<AccC>(lhs_2d[8], rhs, output_2d[8]);
        output_2d[9] = ntt::vmma<AccC>(lhs_2d[9], rhs, output_2d[9]);
        output_2d[10] = ntt::vmma<AccC>(lhs_2d[10], rhs, output_2d[10]);
        output_2d[11] = ntt::vmma<AccC>(lhs_2d[11], rhs, output_2d[11]);
        output_2d[12] = ntt::vmma<AccC>(lhs_2d[12], rhs, output_2d[12]);
        output_2d[13] = ntt::vmma<AccC>(lhs_2d[13], rhs, output_2d[13]);
        output_2d[14] = ntt::vmma<AccC>(lhs_2d[14], rhs, output_2d[14]);
        output_2d[15] = ntt::vmma<AccC>(lhs_2d[15], rhs, output_2d[15]);

        output_2d[16] = ntt::vmma<AccC>(lhs_2d[16], rhs, output_2d[16]);
        output_2d[17] = ntt::vmma<AccC>(lhs_2d[17], rhs, output_2d[17]);
        output_2d[18] = ntt::vmma<AccC>(lhs_2d[18], rhs, output_2d[18]);
        output_2d[19] = ntt::vmma<AccC>(lhs_2d[19], rhs, output_2d[19]);
        output_2d[20] = ntt::vmma<AccC>(lhs_2d[20], rhs, output_2d[20]);
        output_2d[21] = ntt::vmma<AccC>(lhs_2d[21], rhs, output_2d[21]);
        output_2d[22] = ntt::vmma<AccC>(lhs_2d[22], rhs, output_2d[22]);
        output_2d[23] = ntt::vmma<AccC>(lhs_2d[23], rhs, output_2d[23]);

        output_2d[24] = ntt::vmma<AccC>(lhs_2d[24], rhs, output_2d[24]);
        output_2d[25] = ntt::vmma<AccC>(lhs_2d[25], rhs, output_2d[25]);
        output_2d[26] = ntt::vmma<AccC>(lhs_2d[26], rhs, output_2d[26]);
        output_2d[27] = ntt::vmma<AccC>(lhs_2d[27], rhs, output_2d[27]);
        output_2d[28] = ntt::vmma<AccC>(lhs_2d[28], rhs, output_2d[28]);
        output_2d[29] = ntt::vmma<AccC>(lhs_2d[29], rhs, output_2d[29]);
        output_2d[30] = ntt::vmma<AccC>(lhs_2d[30], rhs, output_2d[30]);
        output_2d[31] = ntt::vmma<AccC>(lhs_2d[31], rhs, output_2d[31]);

        output(0) = output_2d[0](0);
        output(1) = output_2d[1](0);
        output(2) = output_2d[2](0);
        output(3) = output_2d[3](0);
        output(4) = output_2d[4](0);
        output(5) = output_2d[5](0);
        output(6) = output_2d[6](0);
        output(7) = output_2d[7](0);

        output(8) = output_2d[8](0);
        output(9) = output_2d[9](0);
        output(10) = output_2d[10](0);
        output(11) = output_2d[11](0);
        output(12) = output_2d[12](0);
        output(13) = output_2d[13](0);
        output(14) = output_2d[14](0);
        output(15) = output_2d[15](0);

        output(16) = output_2d[16](0);
        output(17) = output_2d[17](0);
        output(18) = output_2d[18](0);
        output(19) = output_2d[19](0);
        output(20) = output_2d[20](0);
        output(21) = output_2d[21](0);
        output(22) = output_2d[22](0);
        output(23) = output_2d[23](0);

        output(24) = output_2d[24](0);
        output(25) = output_2d[25](0);
        output(26) = output_2d[26](0);
        output(27) = output_2d[27](0);
        output(28) = output_2d[28](0);
        output(29) = output_2d[29](0);
        output(30) = output_2d[30](0);
        output(31) = output_2d[31](0);

        return output;
    }
};
} // namespace nncase::ntt::vector_ops
