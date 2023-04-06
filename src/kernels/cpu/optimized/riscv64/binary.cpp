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
#include <iostream>
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#if __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

namespace
{

#if __riscv_vector

#define REGISTER_BINARY_OP_RVV(op)                                                             \
    struct binary_op_##op##_rvv                                                                \
    {                                                                                          \
        vfloat32m8_t operator()(const vfloat32m8_t &a, const vfloat32m8_t &b, size_t vl) const \
        {                                                                                      \
            return vf##op##_vv_f32m8(a, b, vl);                                                \
        }                                                                                      \
        vfloat32m8_t operator()(const vfloat32m8_t &a, const float &b, size_t vl) const        \
        {                                                                                      \
            return vf##op##_vf_f32m8(a, b, vl);                                                \
        }                                                                                      \
        vfloat32m8_t operator()(const float &a, const vfloat32m8_t &b, size_t vl) const        \
        {                                                                                      \
            return vf##op##_vf_f32m8(b, a, vl);                                                \
        }                                                                                      \
                                                                                               \
        vint32m8_t operator()(const vint32m8_t &a, const vint32m8_t &b, size_t vl) const       \
        {                                                                                      \
            return v##op##_vv_i32m8(a, b, vl);                                                 \
        }                                                                                      \
        vint32m8_t operator()(const vint32m8_t &a, const int32_t &b, size_t vl) const          \
        {                                                                                      \
            return v##op##_vx_i32m8(a, b, vl);                                                 \
        }                                                                                      \
        vint32m8_t operator()(const int32_t &a, const vint32m8_t &b, size_t vl) const          \
        {                                                                                      \
            return v##op##_vx_i32m8(b, a, vl);                                                 \
        }                                                                                      \
                                                                                               \
        vint64m8_t operator()(const vint64m8_t &a, const vint64m8_t &b, size_t vl) const       \
        {                                                                                      \
            return v##op##_vv_i64m8(a, b, vl);                                                 \
        }                                                                                      \
        vint64m8_t operator()(const vint64m8_t &a, const int64_t &b, size_t vl) const          \
        {                                                                                      \
            return v##op##_vx_i64m8(a, b, vl);                                                 \
        }                                                                                      \
        vint64m8_t operator()(const int64_t &a, const vint64m8_t &b, size_t vl) const          \
        {                                                                                      \
            return v##op##_vx_i64m8(b, a, vl);                                                 \
        }                                                                                      \
    }

REGISTER_BINARY_OP_RVV(add);
REGISTER_BINARY_OP_RVV(mul);
REGISTER_BINARY_OP_RVV(min);
REGISTER_BINARY_OP_RVV(max);

#undef REGISTER_BINARY_OP_RVV

struct binary_op_sub_rvv
{
    // float
    vfloat32m8_t operator()(const vfloat32m8_t &a, const vfloat32m8_t &b, size_t vl) const
    {
        return vfsub_vv_f32m8(a, b, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t &a, const float &b, size_t vl) const
    {
        return vfsub_vf_f32m8(a, b, vl);
    }
    vfloat32m8_t operator()(const float &a, const vfloat32m8_t &b, size_t vl) const
    {
        return vfrsub_vf_f32m8(b, a, vl);
    }

    // int32_t
    vint32m8_t operator()(const vint32m8_t &a, const vint32m8_t &b, size_t vl) const
    {
        return vsub_vv_i32m8(a, b, vl);
    }
    vint32m8_t operator()(const vint32m8_t &a, const int32_t &b, size_t vl) const
    {
        return vsub_vx_i32m8(a, b, vl);
    }
    vint32m8_t operator()(const int32_t &a, const vint32m8_t &b, size_t vl) const
    {
        return vrsub_vx_i32m8(b, a, vl);
    }

    // int64_t
    vint64m8_t operator()(const vint64m8_t &a, const vint64m8_t &b, size_t vl) const
    {
        return vsub_vv_i64m8(a, b, vl);
    }
    vint64m8_t operator()(const vint64m8_t &a, const int64_t &b, size_t vl) const
    {
        return vsub_vx_i64m8(a, b, vl);
    }
    vint64m8_t operator()(const int64_t &a, const vint64m8_t &b, size_t vl) const
    {
        return vrsub_vx_i64m8(b, a, vl);
    }
};

struct binary_op_div_rvv
{
    // float
    vfloat32m8_t operator()(const vfloat32m8_t &a, const vfloat32m8_t &b, size_t vl) const
    {
        return vfdiv_vv_f32m8(a, b, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t &a, const float &b, size_t vl) const
    {
        return vfdiv_vf_f32m8(a, b, vl);
    }
    vfloat32m8_t operator()(const float &a, const vfloat32m8_t &b, size_t vl) const
    {
        return vfrdiv_vf_f32m8(b, a, vl);
    }

    // int32_t
    vint32m8_t operator()(const vint32m8_t &a, const vint32m8_t &b, size_t vl) const
    {
        return vdiv_vv_i32m8(a, b, vl);
    }
    vint32m8_t operator()(const vint32m8_t &a, const int32_t &b, size_t vl) const
    {
        return vdiv_vx_i32m8(a, b, vl);
    }
    vint32m8_t operator()(const int32_t &a, const vint32m8_t &b, size_t vl) const
    {
        return vdiv_vv_i32m8(vmv_v_x_i32m8(a, vl), b, vl);
    }

    // int64_t
    vint64m8_t operator()(const vint64m8_t &a, const vint64m8_t &b, size_t vl) const
    {
        return vdiv_vv_i64m8(a, b, vl);
    }
    vint64m8_t operator()(const vint64m8_t &a, const int64_t &b, size_t vl) const
    {
        return vdiv_vx_i64m8(a, b, vl);
    }
    vint64m8_t operator()(const int64_t &a, const vint64m8_t &b, size_t vl) const
    {
        return vdiv_vv_i64m8(vmv_v_x_i64m8(a, vl), b, vl);
    }
};

// float
template <typename Top>
result<void> optimized_binary_impl(const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_b_shape, const runtime_shape_t &out_shape,
    value_range<float> fused_activation) noexcept
{
    Top op;
    const int count = compute_size(out_shape);
    size_t vl = 0;
    auto out = output;
    printf("a shape size: %d, b shape size: %d\n", (int)in_a_shape.size(), (int)in_b_shape.size());

    if (in_a_shape.size() == in_b_shape.size())
    {
        int n = count;

        // both a and b are tensors
        while (n > 0)
        {
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
    else if (in_a_shape.size() < in_b_shape.size())
    {
        assert(in_a_shape.size() == 1);
        if(in_a_shape.size() != 1)
        {
            printf("error in_a_shape.size() != 1 \n");
        }
        if (in_a_shape[0] == 1)
        {
            // a is scalar, b is tensor
            int n = count;
            while (n > 0)
            {
                vl = vsetvl_e32m8(n);
                auto v_b = vle32_v_f32m8(input_b, vl);
                auto v_out = op(input_a[0], v_b, vl);
                vse32_v_f32m8(out, v_out, vl);

                input_b += vl;
                out += vl;
                n -= vl;
            }
        }
        else
        {
            // a is vector, b is tensor
            assert(in_a_shape[0] == in_b_shape.back());
            size_t nch = count / in_a_shape[0];
            for (size_t w = 0; w < nch; w++)
            {
                int n = in_a_shape[0];
                const float *input_a_tmp = input_a;
                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    auto v_a = vle32_v_f32m8(input_a_tmp, vl);
                    auto v_b = vle32_v_f32m8(input_b, vl);
                    auto v_out = op(v_a, v_b, vl);
                    vse32_v_f32m8(out, v_out, vl);

                    input_a_tmp += vl;
                    input_b += vl;
                    out += vl;
                    n -= vl;
                }
            }
        }
    }
    else
    {
        assert(in_b_shape.size() == 1);
        printf("error in_b_shape.size() != 1 \n");
        if (in_b_shape[0] == 1)
        {
            // a is tensor, b is scalar
            int n = count;
            while (n > 0)
            {
                vl = vsetvl_e32m8(n);
                auto v_a = vle32_v_f32m8(input_a, vl);
                auto v_out = op(v_a, input_b[0], vl);
                vse32_v_f32m8(out, v_out, vl);

                input_a += vl;
                out += vl;
                n -= vl;
            }
        }
        else
        {
            // a is tensor, b is vector
            assert(in_a_shape.back() == in_b_shape[0]);
            size_t nch = count / in_b_shape[0];
            for (size_t w = 0; w < nch; w++)
            {
                int n = in_b_shape[0];
                const float *input_b_tmp = input_b;
                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    auto v_a = vle32_v_f32m8(input_a, vl);
                    auto v_b = vle32_v_f32m8(input_b_tmp, vl);
                    auto v_out = op(v_a, v_b, vl);
                    vse32_v_f32m8(out, v_out, vl);

                    input_a += vl;
                    input_b_tmp += vl;
                    out += vl;
                    n -= vl;
                }
            }
        }
    }

    // fused_activation
    for (int i = 0; i < count; i++)
        output[i] = static_cast<float>(kernels::detail::apply_activation(static_cast<float>(output[i]), fused_activation));

    return ok();
}

// int32_t
template <typename Top>
result<void> optimized_binary_impl(const int32_t *input_a, const int32_t *input_b, int32_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_b_shape, const runtime_shape_t &out_shape,
    value_range<float> fused_activation) noexcept
{
    Top op;
    const int count = compute_size(out_shape);
    size_t vl = 0;
    auto out = output;

    if (in_a_shape.size() == in_b_shape.size())
    {
        int n = count;

        // both a and b are tensors
        while (n > 0)
        {
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
    else if (in_a_shape.size() < in_b_shape.size())
    {
        assert(in_a_shape.size() == 1);
        if (in_a_shape[0] == 1)
        {
            // a is scalar, b is tensor
            int n = count;
            while (n > 0)
            {
                vl = vsetvl_e32m8(n);
                auto v_b = vle32_v_i32m8(input_b, vl);
                auto v_out = op(input_a[0], v_b, vl);
                vse32_v_i32m8(out, v_out, vl);

                input_b += vl;
                out += vl;
                n -= vl;
            }
        }
        else
        {
            // a is vector, b is tensor
            assert(in_a_shape[0] == in_b_shape.back());
            size_t nch = count / in_a_shape[0];
            for (size_t w = 0; w < nch; w++)
            {
                int n = in_a_shape[0];
                const int32_t *input_a_tmp = input_a;
                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    auto v_a = vle32_v_i32m8(input_a_tmp, vl);
                    auto v_b = vle32_v_i32m8(input_b, vl);
                    auto v_out = op(v_a, v_b, vl);
                    vse32_v_i32m8(out, v_out, vl);

                    input_a_tmp += vl;
                    input_b += vl;
                    out += vl;
                    n -= vl;
                }
            }
        }
    }
    else
    {
        assert(in_b_shape.size() == 1);
        if (in_b_shape[0] == 1)
        {
            // a is tensor, b is scalar
            int n = count;
            while (n > 0)
            {
                vl = vsetvl_e32m8(n);
                auto v_a = vle32_v_i32m8(input_a, vl);
                auto v_out = op(v_a, input_b[0], vl);
                vse32_v_i32m8(out, v_out, vl);

                input_a += vl;
                out += vl;
                n -= vl;
            }
        }
        else
        {
            // a is tensor, b is vector
            assert(in_a_shape.back() == in_b_shape[0]);
            size_t nch = count / in_b_shape[0];
            for (size_t w = 0; w < nch; w++)
            {
                int n = in_b_shape[0];
                const int32_t *input_b_tmp = input_b;
                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    auto v_a = vle32_v_i32m8(input_a, vl);
                    auto v_b = vle32_v_i32m8(input_b_tmp, vl);
                    auto v_out = op(v_a, v_b, vl);
                    vse32_v_i32m8(out, v_out, vl);

                    input_a += vl;
                    input_b_tmp += vl;
                    out += vl;
                    n -= vl;
                }
            }
        }
    }

    // fused_activation
    for (int i = 0; i < count; i++)
        output[i] = static_cast<int32_t>(kernels::detail::apply_activation(static_cast<float>(output[i]), fused_activation));

    return ok();
}

// int64_t
template <typename Top>
result<void> optimized_binary_impl(const int64_t *input_a, const int64_t *input_b, int64_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_b_shape, const runtime_shape_t &out_shape,
    value_range<float> fused_activation) noexcept
{
    Top op;
    const int count = compute_size(out_shape);
    size_t vl = 0;
    auto out = output;

    if (in_a_shape.size() == in_b_shape.size())
    {
        int n = count;

        // both a and b are tensors
        while (n > 0)
        {
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
    else if (in_a_shape.size() < in_b_shape.size())
    {
        assert(in_a_shape.size() == 1);
        if (in_a_shape[0] == 1)
        {
            // a is scalar, b is tensor
            int n = count;
            while (n > 0)
            {
                vl = vsetvl_e64m8(n);
                auto v_b = vle64_v_i64m8(input_b, vl);
                auto v_out = op(input_a[0], v_b, vl);
                vse64_v_i64m8(out, v_out, vl);

                input_b += vl;
                out += vl;
                n -= vl;
            }
        }
        else
        {
            // a is vector, b is tensor
            assert(in_a_shape[0] == in_b_shape.back());
            size_t nch = count / in_a_shape[0];
            for (size_t w = 0; w < nch; w++)
            {
                int n = in_a_shape[0];
                const int64_t *input_a_tmp = input_a;
                while (n > 0)
                {
                    vl = vsetvl_e64m8(n);
                    auto v_a = vle64_v_i64m8(input_a_tmp, vl);
                    auto v_b = vle64_v_i64m8(input_b, vl);
                    auto v_out = op(v_a, v_b, vl);
                    vse64_v_i64m8(out, v_out, vl);

                    input_a_tmp += vl;
                    input_b += vl;
                    out += vl;
                    n -= vl;
                }
            }
        }
    }
    else
    {
        assert(in_b_shape.size() == 1);
        if (in_b_shape[0] == 1)
        {
            // a is tensor, b is scalar
            int n = count;
            while (n > 0)
            {
                vl = vsetvl_e64m8(n);
                auto v_a = vle64_v_i64m8(input_a, vl);
                auto v_out = op(v_a, input_b[0], vl);
                vse64_v_i64m8(out, v_out, vl);

                input_a += vl;
                out += vl;
                n -= vl;
            }
        }
        else
        {
            // a is tensor, b is vector
            assert(in_a_shape.back() == in_b_shape[0]);
            size_t nch = count / in_b_shape[0];
            for (size_t w = 0; w < nch; w++)
            {
                int n = in_b_shape[0];
                const int64_t *input_b_tmp = input_b;
                while (n > 0)
                {
                    vl = vsetvl_e64m8(n);
                    auto v_a = vle64_v_i64m8(input_a, vl);
                    auto v_b = vle64_v_i64m8(input_b_tmp, vl);
                    auto v_out = op(v_a, v_b, vl);
                    vse64_v_i64m8(out, v_out, vl);

                    input_a += vl;
                    input_b_tmp += vl;
                    out += vl;
                    n -= vl;
                }
            }
        }
    }

    // fused_activation
    for (int i = 0; i < count; i++)
        output[i] = static_cast<int64_t>(kernels::detail::apply_activation(static_cast<float>(output[i]), fused_activation));

    return ok();
}
#endif
}
#if(0)

template result<void> optimized::binary<float>(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept;
#else
static __inline __attribute__ ((__always_inline__))
void add_f32_vv_r(const float *input0, const float *input1, float *output, int size)
{
    #if(1)
    asm volatile(
                "mv a3, %3;"
                "mv a1, %1;"
                "mv a2, %2;"
                "mv a0, %0;"
                "1:\n\t"
                "vsetvli    t0, a3, e32, m2\n\t"
                "vle32.v      v8, (a1)\n\t"
                "sub        a3, a3, t0\n\t"
                "slli       t0, t0, 2\n\t"      // element: 4 bytes
                "add        a1, a1, t0\n\t"
                "vle32.v      v16, (a2)\n\t"
                "add        a2, a2, t0\n\t"
                "vfadd.vv   v16, v8, v16\n\t"
                "vse32.v      v16, (a0)\n\t"
                "add        a0, a0, t0\n\t"
                "bnez       a3, 1b\n\t"
                :
                :
                "r"(output),
                "r"(input0),
                "r"(input1),
                "r"(size)
                : "v8", "v16", "t0", "t1","a0", "a1", "a2"
    );
    #else
        for(int i = 0; i < size; ++i)
        {
            output[i] = input0[i] + input1[i];
        }
    #endif
}

static __inline __attribute__ ((__always_inline__))
void add_f32_vf(const float *input0, const float input1, float *output, int size)
{
    #if(1)

        asm volatile(
                "mv a3, %3;"
                "mv a1, %1;"
                // "mv ft2, %2;"
                "mv a0, %0;"
                "1:\n\t"
                "vsetvli    t0, a3, e32, m2\n\t"
                "vle32.v      v8, (a1)\n\t"
                "sub        a3, a3, t0\n\t"
                "slli       t0, t0, 2\n\t"      // element: 4 bytes
                "add        a1, a1, t0\n\t"
                "vfadd.vf   v16, v8, %2\n\t"
                "vse32.v      v16, (a0)\n\t"
                "add        a0, a0, t0\n\t"
                "bnez       a3, 1b\n\t"
                :
                :"r"(output),
                "r"(input0),
                "f"(input1),
                "r"(size)
                : "v8", "v16", "t0", "a0", "a1", "a2", "a3"
    );
    #else
        for(int i = 0; i < size; ++i)
        {
            output[i] = input0[i] + input1;
        }
    #endif
}

static uint64_t k230_get_cycles()
{
    uint64_t x;
    __asm volatile("rdcycle %0;"
                   : "=r"(x)::);
    return x;
}
template<> result<void> optimized::binary<float>(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept
    {
        (void)op;(void)input_a;(void)input_b;(void)output;(void)in_a_shape;(void)in_a_strides;(void)in_b_shape;
        (void)in_b_strides;(void)out_shape;(void)out_strides;(void)fused_activation;(void)context;
        if(op != binary_add)
        {
            printf("error op is not add \n");
            return ok();
        }
        // add_f32_vv(input_a, input_b, output, 100);
        int len_a = (int)compute_size(in_a_shape);
        int len_b = (int)compute_size(in_b_shape);
        // int len_out = (int)compute_size(out_shape);
        // float* output_1 = (float*)malloc(len_out * sizeof(float));
        // if(! output_1)
        // {
            // printf("malloc error . \n");
            // return ok();
        // }
        // printf("\n %d, %d, %d \n", len_a, len_b, len_out);
        // printf("before: %f, %f, %f, %f,\n", output[0], output[1], output[2], output[3]);
        // output[0] = 1.0f;output[1] = 2.0f;output[2] = 3.0f;output[3] = 4.0f;
        // printf("after : %f, %f, %f, %f,\n", output[0], output[1], output[2], output[3]);
        // printf("inputa: %f,%f,%f,%f\n", input_a[0], input_a[1], input_a[2], input_a[3]);
        // printf("inputb: %f,%f,%f,%f\n", input_b[0], input_b[1], input_b[2], input_b[3]);
        if(len_a == len_b)
        {
            add_f32_vv_r(input_a, input_b, output, len_a);
            // printf("output_1 : %f, %f, %f, %f,\n", output_1[0], output_1[1], output_1[2], output_1[3]);
            // for(int i = 0; i < len_a; ++i)
            // {
                // //printf("s out:%f, input1:%f, input2:%f\n", output[i], input_a[i], input_b[i]);
                // output[i] = input_a[i] + input_b[i];
                // printf("e out:%f, input1:%f, input2:%f, %f\n", output[i], input_a[i], input_b[i], output_1[i]);
            // }
            
            // printf("output : %f, %f, %f, %f,\n", output[0], output[1], output[2], output[3]);
        }
        else
        {
            if(len_a < len_b)
            {
                add_f32_vf(input_b, input_a[0], output, len_b);
            }
            else
            {
                add_f32_vf(input_a, input_b[0], output, len_a);
            }
        }
        
        // for(int i = 0; i < len_a; ++i)
        // {
            // // printf("error float32 add %f, %f, %d\n", output[i], output_2[i], i);
            // printf("-------------------%f,%f, %f, %f\n", input_a[i], input_b[i], output_1[i], output[i]);
        // }
        #if(1)
        int len_c = (int)compute_size(out_shape);
        float* output_2 = (float*)malloc(len_c * sizeof(float));
        // result<void> rt = cpu::reference::binary(op, input_a, input_b, output_2, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides,
        // fused_activation, context);
        result<void> rt = optimized_binary_impl<binary_op_add_rvv>(input_a, input_b, output, in_a_shape, in_b_shape, out_shape, fused_activation);
        for(int i = 0; i < len_c; ++i)
        {
            if(output[i] != output_2[i])
            {
                printf("error float32 add %f, %f, %f, %f,%d\n", output[i], output_2[i] , input_a[i], input_b[i], i);
                // printf("-------------------%f,%f\n", input_a[i], input_b[i]);
            }
        }
        printf("%d,over... \n", len_c);
        free(output_2);
        return rt;
        #endif
        return ok();
    }
#endif

template result<void> optimized::binary<int32_t>(binary_op_t op, const int32_t *input_a, const int32_t *input_b, int32_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept;

template result<void> optimized::binary<int64_t>(binary_op_t op, const int64_t *input_a, const int64_t *input_b, int64_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept;

template <typename T>
result<void> optimized::binary(binary_op_t op, const T *input_a, const T *input_b, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept
{
    printf("optimized::binary ... \n");
#if __riscv_vector
    printf("optimized::binary ... __riscv_vector\n");
    switch (op)
    {
    case binary_add:
    {
        return optimized_binary_impl<binary_op_add_rvv>(input_a, input_b, output, in_a_shape, in_b_shape, out_shape, fused_activation);
    }
    case binary_sub:
    {
        return optimized_binary_impl<binary_op_sub_rvv>(input_a, input_b, output, in_a_shape, in_b_shape, out_shape, fused_activation);
    }
    case binary_mul:
    {
        return optimized_binary_impl<binary_op_mul_rvv>(input_a, input_b, output, in_a_shape, in_b_shape, out_shape, fused_activation);
    }
    case binary_div:
    {
        return optimized_binary_impl<binary_op_div_rvv>(input_a, input_b, output, in_a_shape, in_b_shape, out_shape, fused_activation);
    }
    case binary_min:
    {
        return optimized_binary_impl<binary_op_min_rvv>(input_a, input_b, output, in_a_shape, in_b_shape, out_shape, fused_activation);
    }
    case binary_max:
    {
        return optimized_binary_impl<binary_op_max_rvv>(input_a, input_b, output, in_a_shape, in_b_shape, out_shape, fused_activation);
    }
    default:
        std::cout << "Unsupported binary op: " + binary_op_to_string(op) + " for optimizing, fallback to reference" << std::endl;
    }
#endif
printf("reference::binary ... \n");
    return cpu::reference::binary(op, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides,
        fused_activation, context);
}