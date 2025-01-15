

#pragma once 
#include "nncase/ntt/arch/riscv64/arch_types.h"
#include "nncase/ntt/vector.h"
#include "../../../half.h"
#include "rvv_mathfun.h"
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif


namespace nncase::ntt::ops{

#ifdef __riscv_vector


#define RVV_UNARY16_OP(op, dtype, vl, kernel)                                    \
    template <> struct op<ntt::vector<dtype, vl>> {                            \
        ntt::vector<dtype, vl>                                                 \
        operator()(const ntt::vector<dtype, vl> &v) const noexcept {           \
            return kernel(v, vl);                                              \
        }                                                                      \
    };

// unary with hlaf
#define REGISTER_RVV_UNARY16_OP(OP, dtype, kernel)                               \
    RVV_UNARY16_OP(OP, half, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)           \
    RVV_UNARY16_OP(OP, half, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)           \
    RVV_UNARY16_OP(OP, half, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)           \
    RVV_UNARY16_OP(OP, half, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

#define ABS_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t abs_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {  \
         return __riscv_vfabs_v_f16m##lmul(v, vl);   \
    }

REGISTER_RVV_KERNEL(ABS_FLOAT16)
REGISTER_RVV_UNARY16_OP(abs, half, abs_float16)


#endif
}
