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
#include "../../reference/ref_ops.h"
#include "../opt_ops.h"
#include <iostream>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#if __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;
namespace {

#if __riscv_vector

#define REGISTER_BINARY_OP_RVV(op)                                             \
    struct binary_op_##op##_rvv {                                              \
        vfloat32m8_t operator()(const vfloat32m8_t &a, const vfloat32m8_t &b,  \
                                size_t vl) const {                             \
            return vf##op##_vv_f32m8(a, b, vl);                                \
        }                                                                      \
        vfloat32m8_t operator()(const vfloat32m8_t &a, const float &b,         \
                                size_t vl) const {                             \
            return vf##op##_vf_f32m8(a, b, vl);                                \
        }                                                                      \
        vfloat32m8_t operator()(const float &a, const vfloat32m8_t &b,         \
                                size_t vl) const {                             \
            return vf##op##_vf_f32m8(b, a, vl);                                \
        }                                                                      \
                                                                               \
        vint32m8_t operator()(const vint32m8_t &a, const vint32m8_t &b,        \
                              size_t vl) const {                               \
            return v##op##_vv_i32m8(a, b, vl);                                 \
        }                                                                      \
        vint32m8_t operator()(const vint32m8_t &a, const int32_t &b,           \
                              size_t vl) const {                               \
            return v##op##_vx_i32m8(a, b, vl);                                 \
        }                                                                      \
        vint32m8_t operator()(const int32_t &a, const vint32m8_t &b,           \
                              size_t vl) const {                               \
            return v##op##_vx_i32m8(b, a, vl);                                 \
        }                                                                      \
                                                                               \
        vint64m8_t operator()(const vint64m8_t &a, const vint64m8_t &b,        \
                              size_t vl) const {                               \
            return v##op##_vv_i64m8(a, b, vl);                                 \
        }                                                                      \
        vint64m8_t operator()(const vint64m8_t &a, const int64_t &b,           \
                              size_t vl) const {                               \
            return v##op##_vx_i64m8(a, b, vl);                                 \
        }                                                                      \
        vint64m8_t operator()(const int64_t &a, const vint64m8_t &b,           \
                              size_t vl) const {                               \
            return v##op##_vx_i64m8(b, a, vl);                                 \
        }                                                                      \
    }

REGISTER_BINARY_OP_RVV(add);
REGISTER_BINARY_OP_RVV(mul);
REGISTER_BINARY_OP_RVV(min);
REGISTER_BINARY_OP_RVV(max);

#undef REGISTER_BINARY_OP_RVV

struct binary_op_sub_rvv {
    // float
    vfloat32m8_t operator()(const vfloat32m8_t &a, const vfloat32m8_t &b,
                            size_t vl) const {
        return vfsub_vv_f32m8(a, b, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t &a, const float &b,
                            size_t vl) const {
        return vfsub_vf_f32m8(a, b, vl);
    }
    vfloat32m8_t operator()(const float &a, const vfloat32m8_t &b,
                            size_t vl) const {
        return vfrsub_vf_f32m8(b, a, vl);
    }

    // int32_t
    vint32m8_t operator()(const vint32m8_t &a, const vint32m8_t &b,
                          size_t vl) const {
        return vsub_vv_i32m8(a, b, vl);
    }
    vint32m8_t operator()(const vint32m8_t &a, const int32_t &b,
                          size_t vl) const {
        return vsub_vx_i32m8(a, b, vl);
    }
    vint32m8_t operator()(const int32_t &a, const vint32m8_t &b,
                          size_t vl) const {
        return vrsub_vx_i32m8(b, a, vl);
    }

    // int64_t
    vint64m8_t operator()(const vint64m8_t &a, const vint64m8_t &b,
                          size_t vl) const {
        return vsub_vv_i64m8(a, b, vl);
    }
    vint64m8_t operator()(const vint64m8_t &a, const int64_t &b,
                          size_t vl) const {
        return vsub_vx_i64m8(a, b, vl);
    }
    vint64m8_t operator()(const int64_t &a, const vint64m8_t &b,
                          size_t vl) const {
        return vrsub_vx_i64m8(b, a, vl);
    }
};

struct binary_op_div_rvv {
    // float
    vfloat32m8_t operator()(const vfloat32m8_t &a, const vfloat32m8_t &b,
                            size_t vl) const {
        return vfdiv_vv_f32m8(a, b, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t &a, const float &b,
                            size_t vl) const {
        return vfdiv_vf_f32m8(a, b, vl);
    }
    vfloat32m8_t operator()(const float &a, const vfloat32m8_t &b,
                            size_t vl) const {
        return vfrdiv_vf_f32m8(b, a, vl);
    }

    // int32_t
    vint32m8_t operator()(const vint32m8_t &a, const vint32m8_t &b,
                          size_t vl) const {
        return vdiv_vv_i32m8(a, b, vl);
    }
    vint32m8_t operator()(const vint32m8_t &a, const int32_t &b,
                          size_t vl) const {
        return vdiv_vx_i32m8(a, b, vl);
    }
    vint32m8_t operator()(const int32_t &a, const vint32m8_t &b,
                          size_t vl) const {
        return vdiv_vv_i32m8(vmv_v_x_i32m8(a, vl), b, vl);
    }

    // int64_t
    vint64m8_t operator()(const vint64m8_t &a, const vint64m8_t &b,
                          size_t vl) const {
        return vdiv_vv_i64m8(a, b, vl);
    }
    vint64m8_t operator()(const vint64m8_t &a, const int64_t &b,
                          size_t vl) const {
        return vdiv_vx_i64m8(a, b, vl);
    }
    vint64m8_t operator()(const int64_t &a, const vint64m8_t &b,
                          size_t vl) const {
        return vdiv_vv_i64m8(vmv_v_x_i64m8(a, vl), b, vl);
    }
};

struct binary_op_mod_rvv {
    // float(fmod)
    vfloat32m8_t operator()(const vfloat32m8_t &a, const vfloat32m8_t &b,
                            size_t vl) const {
        auto quotient_f = vfdiv_vv_f32m8(a, b, vl);
        auto quotient_x = vfcvt_rtz_x_f_v_i32m8(quotient_f, vl);
        quotient_f = vfcvt_f_x_v_f32m8(quotient_x, vl);
        return vfnmsub_vv_f32m8(quotient_f, b, a, vl);
    }

    vfloat32m8_t operator()(const vfloat32m8_t &a, const float &b,
                            size_t vl) const {
        auto quotient_f = vfdiv_vf_f32m8(a, b, vl);
        auto quotient_x = vfcvt_rtz_x_f_v_i32m8(quotient_f, vl);
        quotient_f = vfcvt_f_x_v_f32m8(quotient_x, vl);
        return vfnmsub_vf_f32m8(quotient_f, b, a, vl);
    }

    vfloat32m8_t operator()(const float &a, const vfloat32m8_t &b,
                            size_t vl) const {
        vfloat32m8_t va;
        va = vfmv_v_f_f32m8(a, vl);
        auto quotient_f = vfrdiv_vf_f32m8(b, a, vl);
        auto quotient_x = vfcvt_rtz_x_f_v_i32m8(quotient_f, vl);
        quotient_f = vfcvt_f_x_v_f32m8(quotient_x, vl);
        return vfnmsub_vv_f32m8(quotient_f, b, va, vl);
    }

    // int32_t(floor mod)
    vint32m4_t operator()(const vint32m4_t &a, const vint32m4_t &b,
                          size_t vl) const {
        auto remainder = vrem_vv_i32m4(a, b, vl);
        auto tmp = vxor_vv_i32m4(a, b, vl);
        auto mask1 = vmsne_vx_i32m4_b8(remainder, 0, vl);
        auto mask2 = vmslt_vx_i32m4_b8(tmp, 0, vl);
        mask1 = vmand_mm_b8(mask1, mask2, vl);
        return vadd_vv_i32m4_m(mask1, remainder, remainder, b, vl);
    }

    vint32m4_t operator()(const vint32m4_t &a, const int32_t &b,
                          size_t vl) const {
        auto remainder = vrem_vx_i32m4(a, b, vl);
        auto tmp = vxor_vx_i32m4(a, b, vl);
        auto mask1 = vmsne_vx_i32m4_b8(remainder, 0, vl);
        auto mask2 = vmslt_vx_i32m4_b8(tmp, 0, vl);
        mask1 = vmand_mm_b8(mask1, mask2, vl);
        return vadd_vx_i32m4_m(mask1, remainder, remainder, b, vl);
    }

    vint32m4_t operator()(const int32_t &a, const vint32m4_t &b,
                          size_t vl) const {

        auto va = vmv_v_x_i32m4(a, vl);
        auto remainder = vrem_vv_i32m4(va, b, vl);
        auto tmp = vxor_vv_i32m4(va, b, vl);
        auto mask1 = vmsne_vx_i32m4_b8(remainder, 0, vl);
        auto mask2 = vmslt_vx_i32m4_b8(tmp, 0, vl);
        mask1 = vmand_mm_b8(mask1, mask2, vl);
        return vadd_vv_i32m4_m(mask1, remainder, remainder, b, vl);
    }

    // int64_t(floor mod)
    vint64m4_t operator()(const vint64m4_t &a, const vint64m4_t &b,
                          size_t vl) const {
        auto remainder = vrem_vv_i64m4(a, b, vl);
        auto tmp = vxor_vv_i64m4(a, b, vl);
        auto mask1 = vmsne_vx_i64m4_b16(remainder, 0, vl);
        auto mask2 = vmslt_vx_i64m4_b16(tmp, 0, vl);
        mask1 = vmand_mm_b16(mask1, mask2, vl);
        return vadd_vv_i64m4_m(mask1, remainder, remainder, b, vl);
    }

    vint64m4_t operator()(const vint64m4_t &a, const int64_t &b,
                          size_t vl) const {
        auto remainder = vrem_vx_i64m4(a, b, vl);
        auto tmp = vxor_vx_i64m4(a, b, vl);
        auto mask1 = vmsne_vx_i64m4_b16(remainder, 0, vl);
        auto mask2 = vmslt_vx_i64m4_b16(tmp, 0, vl);
        mask1 = vmand_mm_b16(mask1, mask2, vl);
        return vadd_vx_i64m4_m(mask1, remainder, remainder, b, vl);
    }

    vint64m4_t operator()(const int64_t &a, const vint64m4_t &b,
                          size_t vl) const {
        auto va = vmv_v_x_i64m4(a, vl);
        auto remainder = vrem_vv_i64m4(va, b, vl);
        auto tmp = vxor_vv_i64m4(va, b, vl);
        auto mask1 = vmsne_vx_i64m4_b16(remainder, 0, vl);
        auto mask2 = vmslt_vx_i64m4_b16(tmp, 0, vl);
        mask1 = vmand_mm_b16(mask1, mask2, vl);
        return vadd_vv_i64m4_m(mask1, remainder, remainder, b, vl);
    }
};

// float32
template <typename Top>
void binary_impl_vv_f32(const float *input_a, const float *input_b, float *out,
                        int n) {
    Top op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e32m8(n);
        auto v_a = vle32_v_f32m8(input_a, vl);
        auto v_b = vle32_v_f32m8(input_b, vl);
        auto v_out = op(v_a, v_b, vl);
        vse32_v_f32m8(out, v_out, vl);

        input_a += vl;
        input_b += vl;
        out += vl;
        n -= vl;
    }
}

template <typename Top>
void binary_impl_vf_f32(const float *input_a, float input_b, float *out,
                        int n) {
    Top op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e32m8(n);
        auto v_a = vle32_v_f32m8(input_a, vl);
        auto v_out = op(v_a, input_b, vl);
        vse32_v_f32m8(out, v_out, vl);
        input_a += vl;
        out += vl;
        n -= vl;
    }
}

template <typename Top>
void binary_impl_fv_f32(float input_a, const float *input_b, float *out,
                        int n) {
    Top op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e32m8(n);
        auto v_b = vle32_v_f32m8(input_b, vl);
        auto v_out = op(input_a, v_b, vl);
        vse32_v_f32m8(out, v_out, vl);
        input_b += vl;
        out += vl;
        n -= vl;
    }
}

// int32_t
template <typename Top>
void binary_impl_vv_i32(const int32_t *input_a, const int32_t *input_b,
                        int32_t *out, int n) {
    Top op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e32m8(n);
        auto v_a = vle32_v_i32m8(input_a, vl);
        auto v_b = vle32_v_i32m8(input_b, vl);
        auto v_out = op(v_a, v_b, vl);
        vse32_v_i32m8(out, v_out, vl);

        input_a += vl;
        input_b += vl;
        out += vl;
        n -= vl;
    }
}

template <>
void binary_impl_vv_i32<binary_op_mod_rvv>(const int32_t *input_a,
                                           const int32_t *input_b, int32_t *out,
                                           int n) {
    binary_op_mod_rvv op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e32m4(n);
        auto v_a = vle32_v_i32m4(input_a, vl);
        auto v_b = vle32_v_i32m4(input_b, vl);
        auto v_out = op(v_a, v_b, vl);
        vse32_v_i32m4(out, v_out, vl);

        input_a += vl;
        input_b += vl;
        out += vl;
        n -= vl;
    }
}

template <typename Top>
void binary_impl_vf_i32(const int32_t *input_a, int32_t input_b, int32_t *out,
                        int n) {
    Top op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e32m8(n);
        auto v_a = vle32_v_i32m8(input_a, vl);
        auto v_out = op(v_a, input_b, vl);
        vse32_v_i32m8(out, v_out, vl);
        input_a += vl;
        out += vl;
        n -= vl;
    }
}
template <>
void binary_impl_vf_i32<binary_op_mod_rvv>(const int32_t *input_a,
                                           int32_t input_b, int32_t *out,
                                           int n) {
    binary_op_mod_rvv op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e32m4(n);
        auto v_a = vle32_v_i32m4(input_a, vl);
        auto v_out = op(v_a, input_b, vl);
        vse32_v_i32m4(out, v_out, vl);
        input_a += vl;
        out += vl;
        n -= vl;
    }
}

template <typename Top>
void binary_impl_fv_i32(int32_t input_a, const int32_t *input_b, int32_t *out,
                        int n) {
    Top op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e32m8(n);
        auto v_b = vle32_v_i32m8(input_b, vl);
        auto v_out = op(input_a, v_b, vl);
        vse32_v_i32m8(out, v_out, vl);
        input_b += vl;
        out += vl;
        n -= vl;
    }
}
template <>
void binary_impl_fv_i32<binary_op_mod_rvv>(int32_t input_a,
                                           const int32_t *input_b, int32_t *out,
                                           int n) {
    binary_op_mod_rvv op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e32m4(n);
        auto v_b = vle32_v_i32m4(input_b, vl);
        auto v_out = op(input_a, v_b, vl);
        vse32_v_i32m4(out, v_out, vl);
        input_b += vl;
        out += vl;
        n -= vl;
    }
}

// int64_t
template <typename Top>
void binary_impl_vv_i64(const int64_t *input_a, const int64_t *input_b,
                        int64_t *out, int n) {
    Top op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e64m8(n);
        auto v_a = vle64_v_i64m8(input_a, vl);
        auto v_b = vle64_v_i64m8(input_b, vl);
        auto v_out = op(v_a, v_b, vl);
        vse64_v_i64m8(out, v_out, vl);
        input_a += vl;
        input_b += vl;
        out += vl;
        n -= vl;
    }
}
template <>
void binary_impl_vv_i64<binary_op_mod_rvv>(const int64_t *input_a,
                                           const int64_t *input_b, int64_t *out,
                                           int n) {
    binary_op_mod_rvv op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e64m4(n);
        auto v_a = vle64_v_i64m4(input_a, vl);
        auto v_b = vle64_v_i64m4(input_b, vl);
        auto v_out = op(v_a, v_b, vl);
        vse64_v_i64m4(out, v_out, vl);
        input_a += vl;
        input_b += vl;
        out += vl;
        n -= vl;
    }
}

template <typename Top>
void binary_impl_vf_i64(const int64_t *input_a, int64_t input_b, int64_t *out,
                        int n) {
    Top op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e64m8(n);
        auto v_a = vle64_v_i64m8(input_a, vl);
        auto v_out = op(v_a, input_b, vl);
        vse64_v_i64m8(out, v_out, vl);
        input_a += vl;
        out += vl;
        n -= vl;
    }
}
template <>
void binary_impl_vf_i64<binary_op_mod_rvv>(const int64_t *input_a,
                                           int64_t input_b, int64_t *out,
                                           int n) {
    binary_op_mod_rvv op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e64m4(n);
        auto v_a = vle64_v_i64m4(input_a, vl);
        auto v_out = op(v_a, input_b, vl);
        vse64_v_i64m4(out, v_out, vl);
        input_a += vl;
        out += vl;
        n -= vl;
    }
}

template <typename Top>
void binary_impl_fv_i64(int64_t input_a, const int64_t *input_b, int64_t *out,
                        int n) {
    Top op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e64m8(n);
        auto v_b = vle64_v_i64m8(input_b, vl);
        auto v_out = op(input_a, v_b, vl);
        vse64_v_i64m8(out, v_out, vl);
        input_b += vl;
        out += vl;
        n -= vl;
    }
}
template <>
void binary_impl_fv_i64<binary_op_mod_rvv>(int64_t input_a,
                                           const int64_t *input_b, int64_t *out,
                                           int n) {
    binary_op_mod_rvv op;
    size_t vl;
    while (n > 0) {
        vl = vsetvl_e64m4(n);
        auto v_b = vle64_v_i64m4(input_b, vl);
        auto v_out = op(input_a, v_b, vl);
        vse64_v_i64m4(out, v_out, vl);
        input_b += vl;
        out += vl;
        n -= vl;
    }
}

static int verify_shape_impl(gsl::span<const size_t> in_a_shape,
                             gsl::span<const size_t> in_b_shape,
                             int *outter_front_size_ptr,
                             int *outter_current_size_ptr) {
    int size_diff = in_a_shape.size() - in_b_shape.size();
    int outter_front_size = 1;
    int outter_current_size = 1;
    for (int i = 0; i < size_diff; ++i) {
        outter_front_size *= in_a_shape[i];
    }
    int index = -1;
    for (int i = 0; i < (int)(in_b_shape.size()); ++i) {
        if (in_b_shape[i] == in_a_shape[i + size_diff]) {
            outter_current_size *= in_b_shape[i];
            index = i;
        } else {
            break;
        }
    }

    if (index == (int)(in_b_shape.size() - 1)) {
        return index;
    }

    if (in_b_shape[in_b_shape.size() - 1] == 1) {
        int len_b_leave = 1;
        for (int i = index + 1; i < (int)in_b_shape.size(); ++i) {
            len_b_leave *= in_b_shape[i];
        }
        if (len_b_leave != 1) {
            printf("error ... in_b_shape_ptr[i] != 1");
            return -1;
        }
        *outter_front_size_ptr = outter_front_size;
        *outter_current_size_ptr = outter_current_size;
        return index;
    }
    return -1;
}

static int verify_shape(gsl::span<const size_t> in_a_shape,
                        gsl::span<const size_t> in_b_shape, int a_len,
                        int b_len, gsl::span<const size_t> out_shape,
                        int *outter_front_size, int *outter_current_size) {
    if ((in_a_shape != out_shape) && (in_b_shape != out_shape)) {
        return -1;
    }
    if (a_len == 1 || b_len == 1)
        return 0;
    if (a_len > b_len) {
        return verify_shape_impl(in_a_shape, in_b_shape, outter_front_size,
                                 outter_current_size);
    }
    if (a_len < b_len) {
        return verify_shape_impl(in_b_shape, in_a_shape, outter_front_size,
                                 outter_current_size);
    }
    if (in_a_shape.size() < in_b_shape.size()) {
        return verify_shape_impl(in_b_shape, in_a_shape, outter_front_size,
                                 outter_current_size);
    }
    return verify_shape_impl(in_a_shape, in_b_shape, outter_front_size,
                             outter_current_size);
}

static gsl::span<const size_t>
get_sample_span(gsl::span<const size_t> in_shape) {
    int not_one_index = 0;
    for (int i = 0; i < (int)in_shape.size(); ++i) {
        if (in_shape[i] != 1) {
            not_one_index = i;
            break;
        }
    }
    if (not_one_index != 0) {
        return gsl::span<const size_t>(in_shape.begin() + not_one_index,
                                       in_shape.end());
    } else {
        return in_shape;
    }
}

#define REGISTER_BINARY_IMPL(data_type, function_ptr_vs, function_ptr_sv,      \
                             function_ptr_vv)                                  \
    template <typename Top>                                                    \
    int optimized_binary_impl(                                                 \
        const data_type *input_a, const data_type *input_b, data_type *output, \
        gsl::span<const size_t> in_a_shape,                                    \
        gsl::span<const size_t> in_b_shape,                                    \
        [[maybe_unused]] gsl::span<const size_t> out_shape) noexcept {         \
        in_a_shape = get_sample_span(in_a_shape);                              \
        in_b_shape = get_sample_span(in_b_shape);                              \
        out_shape = get_sample_span(out_shape);                                \
        int len_a =                                                            \
            in_a_shape.size() != 0 ? (int)compute_size(in_a_shape) : 1;        \
        int len_b =                                                            \
            in_b_shape.size() != 0 ? (int)compute_size(in_b_shape) : 1;        \
        if (in_a_shape == in_b_shape) {                                        \
            function_ptr_vv<Top>(input_a, input_b, output, len_a);             \
            return 0;                                                          \
        }                                                                      \
        int outter_front_size, outter_current_size;                            \
        int index =                                                            \
            verify_shape(in_a_shape, in_b_shape, len_a, len_b, out_shape,      \
                         &outter_front_size, &outter_current_size);            \
        if (index == -1)                                                       \
            return -1;                                                         \
        if (len_a >= len_b) {                                                  \
            if (len_b == 1) {                                                  \
                function_ptr_vs<Top>(input_a, input_b[0], output, len_a);      \
            } else {                                                           \
                if ((in_b_shape[in_b_shape.size() - 1] == 1) &&                \
                    (index != (int)(in_b_shape.size() - 1))) {                 \
                    int size_diff = in_a_shape.size() - in_b_shape.size();     \
                    int len_a_leave = 1;                                       \
                    for (int i = index + 1; i < (int)in_b_shape.size(); ++i) { \
                        len_a_leave *= in_a_shape[i + size_diff];              \
                    }                                                          \
                    for (int j = 0; j < outter_front_size; ++j) {              \
                        for (int i = 0; i < outter_current_size; ++i) {        \
                            function_ptr_vs<Top>(                              \
                                input_a + len_a_leave * i +                    \
                                    j * len_a_leave * outter_current_size,     \
                                input_b[i],                                    \
                                output + len_a_leave * i +                     \
                                    j * len_a_leave * outter_current_size,     \
                                len_a_leave);                                  \
                        }                                                      \
                    }                                                          \
                } else {                                                       \
                    int loop_n = len_a / len_b;                                \
                    for (int i = 0; i < loop_n; ++i) {                         \
                        function_ptr_vv<Top>(input_a, input_b, output, len_b); \
                        input_a += len_b;                                      \
                        output += len_b;                                       \
                    }                                                          \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            if (len_a == 1) {                                                  \
                function_ptr_sv<Top>(input_a[0], input_b, output, len_b);      \
            } else {                                                           \
                if ((in_a_shape[in_a_shape.size() - 1] == 1) &&                \
                    (index != (int)(in_a_shape.size() - 1))) {                 \
                    int size_diff = in_b_shape.size() - in_a_shape.size();     \
                    int len_b_leave = 1;                                       \
                    for (int i = index + 1; i < (int)in_a_shape.size(); ++i) { \
                        len_b_leave *= in_b_shape[i + size_diff];              \
                    }                                                          \
                    for (int j = 0; j < outter_front_size; ++j) {              \
                        for (int i = 0; i < outter_current_size; ++i) {        \
                            function_ptr_sv<Top>(                              \
                                input_a[i],                                    \
                                input_b + len_b_leave * i +                    \
                                    j * len_b_leave * outter_current_size,     \
                                output + len_b_leave * i +                     \
                                    j * len_b_leave * outter_current_size,     \
                                len_b_leave);                                  \
                        }                                                      \
                    }                                                          \
                } else {                                                       \
                    int loop_n = len_b / len_a;                                \
                    for (int i = 0; i < loop_n; ++i) {                         \
                        function_ptr_vv<Top>(input_a, input_b, output, len_a); \
                        input_b += len_a;                                      \
                        output += len_a;                                       \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        return 0;                                                              \
    }

REGISTER_BINARY_IMPL(float, binary_impl_vf_f32, binary_impl_fv_f32,
                     binary_impl_vv_f32)
REGISTER_BINARY_IMPL(int32_t, binary_impl_vf_i32, binary_impl_fv_i32,
                     binary_impl_vv_i32)
REGISTER_BINARY_IMPL(int64_t, binary_impl_vf_i64, binary_impl_fv_i64,
                     binary_impl_vv_i64)

#endif
} // namespace

result<void> optimized::binary(
    typecode_t typecode, runtime::stackvm::binary_op_t op, const gsl::byte *lhs,
    const gsl::byte *rhs, gsl::byte *out, gsl::span<const size_t> in_a_shape,
    gsl::span<const size_t> lhs_strides, gsl::span<const size_t> in_b_shape,
    gsl::span<const size_t> rhs_strides, gsl::span<const size_t> out_shape,
    gsl::span<const size_t> out_strides,
    NNCASE_UNUSED kernel_context &context) noexcept {
    int ret_value = -1;
#if __riscv_vector
#define BINARY_IMPL(_ty)                                                       \
    {                                                                          \
        auto *input_a = IN_CAST(_ty, lhs);                                     \
        auto *input_b = IN_CAST(_ty, rhs);                                     \
        auto *output = OUT_CAST(_ty, out);                                     \
        switch (op) {                                                          \
        case binary_op_t::add: {                                               \
            ret_value = optimized_binary_impl<binary_op_add_rvv>(              \
                input_a, input_b, output, in_a_shape, in_b_shape, out_shape);  \
            break;                                                             \
        }                                                                      \
        case binary_op_t::sub: {                                               \
            ret_value = optimized_binary_impl<binary_op_sub_rvv>(              \
                input_a, input_b, output, in_a_shape, in_b_shape, out_shape);  \
            break;                                                             \
        }                                                                      \
        case binary_op_t::mul: {                                               \
            ret_value = optimized_binary_impl<binary_op_mul_rvv>(              \
                input_a, input_b, output, in_a_shape, in_b_shape, out_shape);  \
            break;                                                             \
        }                                                                      \
        case binary_op_t::div: {                                               \
            ret_value = optimized_binary_impl<binary_op_div_rvv>(              \
                input_a, input_b, output, in_a_shape, in_b_shape, out_shape);  \
            break;                                                             \
        }                                                                      \
        case binary_op_t::mod: {                                               \
            ret_value = optimized_binary_impl<binary_op_mod_rvv>(              \
                input_a, input_b, output, in_a_shape, in_b_shape, out_shape);  \
            break;                                                             \
        }                                                                      \
        case binary_op_t::min: {                                               \
            ret_value = optimized_binary_impl<binary_op_min_rvv>(              \
                input_a, input_b, output, in_a_shape, in_b_shape, out_shape);  \
            break;                                                             \
        }                                                                      \
        case binary_op_t::max: {                                               \
            ret_value = optimized_binary_impl<binary_op_max_rvv>(              \
                input_a, input_b, output, in_a_shape, in_b_shape, out_shape);  \
            break;                                                             \
        }                                                                      \
        default:                                                               \
            ret_value = -1;                                                    \
            break;                                                             \
        }                                                                      \
        break;                                                                 \
    }

    switch (typecode) {
    case dt_float32:
        BINARY_IMPL(float);
    case dt_int32:
        BINARY_IMPL(int32_t);
    case dt_int64:
        BINARY_IMPL(int64_t);
    default:;
    }
#endif
    if (!ret_value) {
        return ok();
    }

    return stackvm::reference::binary(typecode, op, lhs, rhs, out, in_a_shape,
                                      lhs_strides, in_b_shape, rhs_strides,
                                      out_shape, out_strides, context);
}