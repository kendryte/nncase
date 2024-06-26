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
#include "utils.h"
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;
using namespace nncase::runtime::stackvm;

namespace {
#if __riscv_vector

struct unary_op_abs_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return __riscv_vfabs_v_f32m8(x, vl);
    }
};

struct unary_op_ceil_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        vint32m8_t _xi = __riscv_vfcvt_x_f_v_i32m8(x, vl);
        vbool4_t _mask = __riscv_vmflt_vv_f32m8_b4(
            __riscv_vfcvt_f_x_v_f32m8(_xi, vl), x, vl);
        return __riscv_vfcvt_f_x_v_f32m8(
            __riscv_vadd_vx_i32m8_m(_mask, _xi, 1, vl), vl);
    }
};

struct unary_op_cos_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return cos_ps(x, vl);
    }
};

struct unary_op_exp_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return exp_ps(x, vl);
    }
};

struct unary_op_floor_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
#if 1
        vint32m8_t _xi = __riscv_vfcvt_x_f_v_i32m8(x, vl);
        vbool4_t _mask = __riscv_vmfgt_vv_f32m8_b4(
            __riscv_vfcvt_f_x_v_f32m8(_xi, vl), x, vl);
        return __riscv_vfcvt_f_x_v_f32m8(
            __riscv_vsub_vx_i32m8_m(_mask, _xi, 1, vl), vl);
#else
        float f = 0.5f;
        auto tmp = __riscv_vfsub_vf_f32m8(x, f, vl);
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8(tmp, vl),
                                         vl);
#endif
    }
};

struct unary_op_log_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return log_ps(x, vl);
    }
};

struct unary_op_neg_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return __riscv_vfneg_v_f32m8(x, vl);
    }
};

struct unary_op_round_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8(x, vl), vl);
    }
};

struct unary_op_rsqrt_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        float one = 1.f;
        return __riscv_vfrdiv_vf_f32m8(__riscv_vfsqrt_v_f32m8(x, vl), one, vl);
    }
};

// sign(x) = (x > 0) - (x < 0)
struct unary_op_sign_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        float zero = 0.f;
        auto zeros = __riscv_vfmv_v_f_f32m8(zero, vl);
        float one = 1.f;
        auto gt_mask = __riscv_vmfgt_vf_f32m8_b4(x, zero, vl);
        auto ret = __riscv_vfadd_vf_f32m8_m(gt_mask, zeros, one, vl);
        auto lt_mask = __riscv_vmflt_vf_f32m8_b4(x, zero, vl);
        ret = __riscv_vfsub_vf_f32m8_m(lt_mask, ret, one, vl);

        return ret;
    }
};

struct unary_op_sin_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return sin_ps(x, vl);
    }
};

struct unary_op_sqrt_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return __riscv_vfsqrt_v_f32m8(x, vl);
    }
};

struct unary_op_square_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return __riscv_vfmul_vv_f32m8(x, x, vl);
    }
};

struct unary_op_tanh_rvv {
    vfloat32m8_t operator()(const vfloat32m8_t &x, const size_t &vl) const {
        return tanh_ps(x, vl);
    }
};

// float
template <typename Top>
result<void> optimized_unary_impl(const float *input, float *output,
                                  gsl::span<const size_t> shape) noexcept {
    Top op;
    int32_t n = compute_size(shape);
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m8(n);
        auto v_in = __riscv_vle32_v_f32m8(input, vl);
        auto v_out = op(v_in, vl);
        __riscv_vse32_v_f32m8(output, v_out, vl);

        input += vl;
        output += vl;
        n -= vl;
    }

    return ok();
}
#endif
} // namespace

// result<void> optimized::unary(runtime::stackvm::unary_op_t op, const float
// *input, float *output, gsl::span<const size_t> shape,
//    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
//    kernel_context &context) noexcept
result<void> optimized::unary(typecode_t dtype, runtime::stackvm::unary_op_t op,
                              const gsl::byte *in, gsl::byte *out,
                              gsl::span<const size_t> shape,
                              gsl::span<const size_t> in_strides,
                              gsl::span<const size_t> out_shape,
                              gsl::span<const size_t> out_strides,
                              kernel_context &context) noexcept {
    if (dtype == dt_float32) {
#if __riscv_vector
        auto *input = IN_CAST(float, in);
        auto *output = OUT_CAST(float, out);
        switch (op) {
        case unary_op_t::abs: {
            return optimized_unary_impl<unary_op_abs_rvv>(input, output, shape);
        }
        case unary_op_t::ceil: {
            return optimized_unary_impl<unary_op_ceil_rvv>(input, output,
                                                           shape);
        }
        case unary_op_t::cos: {
            return optimized_unary_impl<unary_op_cos_rvv>(input, output, shape);
        }
        case unary_op_t::exp: {
            return optimized_unary_impl<unary_op_exp_rvv>(input, output, shape);
        }
        case unary_op_t::floor: {
            return optimized_unary_impl<unary_op_floor_rvv>(input, output,
                                                            shape);
        }
        case unary_op_t::log: {
            return optimized_unary_impl<unary_op_log_rvv>(input, output, shape);
        }
        case unary_op_t::neg: {
            return optimized_unary_impl<unary_op_neg_rvv>(input, output, shape);
        }
        case unary_op_t::round: {
            return optimized_unary_impl<unary_op_round_rvv>(input, output,
                                                            shape);
        }
        case unary_op_t::rsqrt: {
            return optimized_unary_impl<unary_op_rsqrt_rvv>(input, output,
                                                            shape);
        }
        case unary_op_t::sign: {
            return optimized_unary_impl<unary_op_sign_rvv>(input, output,
                                                           shape);
        }
        case unary_op_t::sin: {
            return optimized_unary_impl<unary_op_sin_rvv>(input, output, shape);
        }
        case unary_op_t::sqrt: {
            return optimized_unary_impl<unary_op_sqrt_rvv>(input, output,
                                                           shape);
        }
        case unary_op_t::square: {
            return optimized_unary_impl<unary_op_square_rvv>(input, output,
                                                             shape);
        }
        case unary_op_t::tanh: {
            return optimized_unary_impl<unary_op_tanh_rvv>(input, output,
                                                           shape);
        }
        default:;
            //        std::cout << "Unsupported unary op: " +
            //        unary_op_to_string(op)
            //        + " for optimizing, fallback to reference" << std::endl;
        }
#endif
    }
    return stackvm::reference::unary(dtype, op, in, out, shape, in_strides,
                                     out_shape, out_strides, context);
}