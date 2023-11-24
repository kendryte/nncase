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
#pragma once
#include <cstring>
#if __riscv_vector
#include "riscv64/utils.h"
#define _STR(x) #x
#define STR(x) _STR(x)
#define _CONNECT(a, b) a##b
#define CONNECT(a, b) _CONNECT(a, b)
#define vsetvli_macro(evl, avl, elen, mlen)  \
    "vsetvli " STR(evl) "," STR(avl) "," STR(CONNECT(e, elen)) "," STR(CONNECT(m, mlen)) ";"
#define vle_len_macro(eew,vd, rs) \
STR(CONNECT(vle, eew)) ".v" " " STR(vd) "," STR(rs) ";"

#define vse_len_macro(eew,vd, rs) \
STR(CONNECT(vse, eew)) ".v" " " STR(vd) "," STR(rs) ";"

#define vluxei_len_macro(ilen,vd, rs, vindex) \
STR(CONNECT(vluxei, ilen)) ".v" " " STR(vd) "," STR(rs) "," STR(vindex) ";"

#define vsllvi_len_macro(vd,vsrc,shift_bits) \
"vsll.vi " STR(vd) ", " STR(vsrc) ", " STR(shift_bits) ";"

#define slli_len_macro(rd,rs,shift_bits) \
"slli " STR(rd) ", " STR(rs) ", " STR(shift_bits) ";"

#define srli_len_macro(rd,rs,shift_bits) \
"srli " STR(rd) ", " STR(rs) ", " STR(shift_bits) ";"

#define addi_macro(rd,rs,add_num) \
"addi " STR(rd) ", " STR(rs) ", " STR(add_num) ";"

#define vsse_len_macro(ilen,vd, md, stride) \
STR(CONNECT(vsse, ilen)) ".v" " " STR(vd) "," STR(md) "," STR(stride) ";"

#define vaddi_macro(vd, vs, idata) \
"vadd.vi " STR(vd) ", " STR(vs) ", " STR(idata) ";"
#define date_type_bits 32
#define bit_shift 2
#define emul 8
static void *cy_data(void *dst, const void *src, int data_bytes)
{
    __asm volatile(
        "mv a0, %[data_bytes];"
        "mv a1, %[src];"
        "mv a2, %[dst];" srli_len_macro(
            a0, a0, bit_shift) "loop1cpy_data%=:;" vsetvli_macro(t0, a0,
                                                                 date_type_bits,
                                                                 emul)
            vle_len_macro(date_type_bits, v8, (a1))
                slli_len_macro(t1, t0, bit_shift) vse_len_macro(
                    date_type_bits, v8, (a2)) "add a1, a1, t1;"
                                              "add a2, a2, t1;"
                                              "sub a0, a0, t0;"
                                              "bnez a0, loop1cpy_data%=;"
        :
        : [src] "r"(src), [data_bytes] "r"(data_bytes), [dst] "r"(dst)
        : "t0", "t1", "a0", "a1", "a2", "v8", "v16");
    return dst
}
#endif

// todo: reimplement memcpy with rvv
inline void *opt_memcpy(void *dst, const void *src, size_t n) {
#if __riscv_vector
    return cy_data(dst, src, n);
#else
    return memcpy(dst, src, n);
#endif
}