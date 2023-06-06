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

static int verify_shape(dims_t in_a_shape, dims_t in_b_shape) {
    int size_diff = in_a_shape.size() - in_b_shape.size();
    if (size_diff < 0)
        return -1;
    int outter_front_size = 1;
    int outter_current_size = 1;
    for (int i = 0; i < size_diff; ++i) {
        outter_front_size *= in_a_shape[i];
    }
    int index = -1;
    for (int i = 0; i < (int)(in_b_shape.size()); ++i) {
        if (in_b_shape[i] == 1) {
            outter_front_size *= in_a_shape[i + size_diff];
            continue;
        }

        if (in_b_shape[i] == in_a_shape[i + size_diff]) {
            outter_current_size *= in_b_shape[i];
            index = i;
        } else {
            break;
        }
    }

    if (index == (int)(in_b_shape.size() - 1)) {
        return 0;
    }
    return -1;
}

// float
template <typename Top>
int optimized_binary_impl(const float *input_a, const float *input_b,
                          float *output, const dims_t &in_a_shape,
                          const dims_t &in_b_shape,
                          const dims_t &out_shape) noexcept {
    (void)out_shape;
    if (in_a_shape.size() == 0 || in_b_shape.size() == 0) {
        return 0;
    }
    int len_a = (int)compute_size(in_a_shape);
    int len_b = (int)compute_size(in_b_shape);
    if (in_a_shape == in_b_shape) {
        binary_impl_vv_f32<Top>(input_a, input_b, output, len_a);
        return 0;
    }
    if (len_a >= len_b) {
        if (verify_shape(in_a_shape, in_b_shape)) // 校验失败
        {
            return -1;
        }
        if (len_b == 1) {
            binary_impl_vf_f32<Top>(input_a, input_b[0], output, len_a);
        } else {
            int loop_n = len_a / len_b;
            for (int i = 0; i < loop_n; ++i) {
                binary_impl_vv_f32<Top>(input_a, input_b, output, len_b);
                input_a += len_b;
                output += len_b;
            }
        }
    } else // len_a < len_b
    {
        if (verify_shape(in_b_shape, in_a_shape)) // 校验失败
        {
            return -1;
        }
        if (len_a == 1) {
            binary_impl_fv_f32<Top>(input_a[0], input_b, output, len_b);
        } else {
            int loop_n = len_b / len_a;
            for (int i = 0; i < loop_n; ++i) {
                binary_impl_vv_f32<Top>(input_a, input_b, output, len_a);
                input_b += len_a;
                output += len_a;
            }
        }
    }

    return 0;
}

// int32_t
template <typename Top>
int optimized_binary_impl(const int32_t *input_a, const int32_t *input_b,
                          int32_t *output, const dims_t &in_a_shape,
                          const dims_t &in_b_shape,
                          const dims_t &out_shape) noexcept {
    (void)out_shape;
    if (in_a_shape.size() == 0 || in_b_shape.size() == 0) {
        return 0;
    }
    int len_a = (int)compute_size(in_a_shape);
    int len_b = (int)compute_size(in_b_shape);
    if (in_a_shape == in_b_shape) {
        binary_impl_vv_i32<Top>(input_a, input_b, output, len_a);
        return 0;
    }
    if (len_a >= len_b) {
        if (verify_shape(in_a_shape, in_b_shape)) // 校验失败
        {
            return -1;
        }
        if (len_b == 1) {
            binary_impl_vf_i32<Top>(input_a, input_b[0], output, len_a);
        } else {
            int loop_n = len_a / len_b;
            for (int i = 0; i < loop_n; ++i) {
                binary_impl_vv_i32<Top>(input_a, input_b, output, len_b);
                input_a += len_b;
                output += len_b;
            }
        }
    } else // len_a < len_b
    {
        if (verify_shape(in_b_shape, in_a_shape)) // 校验失败
        {
            return -1;
        }
        if (len_a == 1) {
            binary_impl_fv_i32<Top>(input_a[0], input_b, output, len_b);
        } else {
            int loop_n = len_b / len_a;
            for (int i = 0; i < loop_n; ++i) {
                binary_impl_vv_i32<Top>(input_a, input_b, output, len_a);
                input_b += len_a;
                output += len_a;
            }
        }
    }

    return 0;
}

// int64_t
template <typename Top>
int optimized_binary_impl(const int64_t *input_a, const int64_t *input_b,
                          int64_t *output, const dims_t &in_a_shape,
                          const dims_t &in_b_shape,
                          const dims_t &out_shape) noexcept {
    (void)out_shape;
    if (in_a_shape.size() == 0 || in_b_shape.size() == 0) {
        return 0;
    }
    int len_a = (int)compute_size(in_a_shape);
    int len_b = (int)compute_size(in_b_shape);
    if (in_a_shape == in_b_shape) {
        binary_impl_vv_i64<Top>(input_a, input_b, output, len_a);
        return 0;
    }
    if (len_a >= len_b) {
        if (verify_shape(in_a_shape, in_b_shape)) // 校验失败
        {
            return -1;
        }
        if (len_b == 1) {
            binary_impl_vf_i64<Top>(input_a, input_b[0], output, len_a);
        } else {
            int loop_n = len_a / len_b;
            for (int i = 0; i < loop_n; ++i) {
                binary_impl_vv_i64<Top>(input_a, input_b, output, len_b);
                input_a += len_b;
                output += len_b;
            }
        }
    } else // len_a < len_b
    {
        if (verify_shape(in_b_shape, in_a_shape)) // 校验失败
        {
            return -1;
        }
        if (len_a == 1) {
            binary_impl_fv_i64<Top>(input_a[0], input_b, output, len_b);
        } else {
            int loop_n = len_b / len_a;
            for (int i = 0; i < loop_n; ++i) {
                binary_impl_vv_i64<Top>(input_a, input_b, output, len_a);
                input_b += len_a;
                output += len_a;
            }
        }
    }

    return 0;
}
#endif
} // namespace

result<void>
optimized::binary(typecode_t typecode, runtime::stackvm::binary_op_t op,
                  const gsl::byte *lhs, const gsl::byte *rhs, gsl::byte *out,
                  const dims_t &in_a_shape, const strides_t &lhs_strides,
                  const dims_t &in_b_shape, const strides_t &rhs_strides,
                  const dims_t &out_shape, const strides_t &out_strides,
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