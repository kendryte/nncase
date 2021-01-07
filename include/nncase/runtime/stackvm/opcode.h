/* Copyright 2020 Canaan Inc.
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
#include "../datatypes.h"

namespace knn::runtime::k510::dsp
{
enum datatype_t
{
    DT_INT8 = 0x00,
    DT_INT32 = 0x01,
    DT_UINT8 = 0x02,
    DT_UINT32 = 0x03,
    DT_BF16 = 0x04,
    DT_FP32 = 0x05
};

template <datatype_t>
struct value_type;

template <>
struct value_type<DT_INT8>
{
    using type = int8_t;
};
template <>
struct value_type<DT_INT32>
{
    using type = int32_t;
};
template <>
struct value_type<DT_UINT8>
{
    using type = uint8_t;
};
template <>
struct value_type<DT_UINT32>
{
    using type = uint32_t;
};
template <>
struct value_type<DT_BF16>
{
    using type = bfloat16;
};
template <>
struct value_type<DT_FP32>
{
    using type = float;
};

template <datatype_t Type>
using value_type_t = typename value_type<Type>::type;

inline size_t get_bytes(datatype_t type) noexcept
{
    switch (type)
    {
    case DT_INT8:
    case DT_UINT8:
        return 1;
    case DT_BF16:
        return 2;
    case DT_INT32:
    case DT_UINT32:
    case DT_FP32:
        return 4;
    default:
        return 0;
    }
}

struct runtime_index_t
{
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;

    uint32_t &operator[](size_t i) noexcept
    {
        switch (i)
        {
        case 0:
            return n;
        case 1:
            return c;
        case 2:
            return h;
        case 3:
            return w;
        default:
            assert(!"Invalid index");
            return w;
        }
    }
};

struct runtime_stride_t
{
    uint32_t c_stride;
    uint32_t h_stride;
    uint32_t w_stride;
};

struct runtime_shape_t
{
    uint32_t n : 32;
    uint32_t c : 32;
    uint32_t h : 32;
    uint32_t w : 32;

    uint32_t c_stride : 32;
    uint32_t h_stride : 32;
    uint32_t w_stride : 32;

    runtime_stride_t stride() const noexcept
    {
        return { c_stride, h_stride, w_stride };
    }

    constexpr uint32_t get(size_t i) const noexcept
    {
        switch (i)
        {
        case 0:
            return n;
        case 1:
            return c;
        case 2:
            return h;
        case 3:
            return w;
        default:
            return 1;
        }
    }

    constexpr void set(size_t i, uint32_t value) noexcept
    {
        switch (i)
        {
        case 0:
            n = value;
            break;
        case 1:
            c = value;
            break;
        case 2:
            h = value;
            break;
        case 3:
            w = value;
            break;
        default:
            break;
        }
    }
};

struct padding_config_dim_t
{
    uint16_t low : 12;
    uint16_t high : 12;
    uint16_t interior : 8;
};

struct slice_config_dim_t
{
    uint32_t start : 32;
    uint32_t end : 32;
    int32_t stride : 32;
};

struct perm_t
{
    uint8_t d0 : 2;
    uint8_t d1 : 2;
    uint8_t d2 : 2;
    uint8_t d3 : 2;
};

enum DSP_OPCODE
{
#define DEFINE_OPCODE(name, value, unused1) name = value,
#include "gnne_dsp_opcode.def"
#undef DEFINE_OPCODE
};

#pragma pack(push, 1)

struct nop
{
    uint8_t opcode = DSP_OPCODE::NOP;
};

struct ldc_i4
{
    uint8_t opcode = DSP_OPCODE::LDC_I4;
    int32_t value;

    ldc_i4(int32_t value)
        : value(value) { }
};

struct ldc_r4
{
    uint8_t opcode = DSP_OPCODE::LDC_R4;
    float value;

    ldc_r4(float value)
        : value(value) { }
};

struct ldind_i1
{
    uint8_t opcode = DSP_OPCODE::LDIND_I1;
    int32_t address;

    ldind_i1(int32_t address)
        : address(address) { }
};

struct ldind_i4
{
    uint8_t opcode = DSP_OPCODE::LDIND_I4;
    int32_t address;

    ldind_i4(int32_t address)
        : address(address) { }
};

struct ldind_u1
{
    uint8_t opcode = DSP_OPCODE::LDIND_U1;
    int32_t address;

    ldind_u1(int32_t address)
        : address(address) { }
};

struct ldind_br2
{
    uint8_t opcode = DSP_OPCODE::LDIND_BR2;
    int32_t address;

    ldind_br2(int32_t address)
        : address(address) { }
};

struct ldind_r4
{
    uint8_t opcode = DSP_OPCODE::LDIND_R4;
    int32_t address;

    ldind_r4(int32_t address)
        : address(address) { }
};

struct stind_i1
{
    uint8_t opcode = DSP_OPCODE::STIND_I1;
    int32_t address;

    stind_i1(int32_t address)
        : address(address) { }
};

struct stind_i4
{
    uint8_t opcode = DSP_OPCODE::STIND_I4;
    int32_t address;

    stind_i4(int32_t address)
        : address(address) { }
};

struct stind_br2
{
    uint8_t opcode = DSP_OPCODE::STIND_BR2;
    int32_t address;

    stind_br2(int32_t address)
        : address(address) { }
};

struct stind_r4
{
    uint8_t opcode = DSP_OPCODE::STIND_R4;
    int32_t address;

    stind_r4(int32_t address)
        : address(address) { }
};

struct lda_s
{
    uint8_t opcode = DSP_OPCODE::LDA_S;
    int32_t reg : 8;
    int32_t offset : 24;

    lda_s(uint8_t reg, int32_t address)
        : reg(reg), offset(address) { }
};

struct dup
{
    uint8_t opcode = DSP_OPCODE::DUP;
};

struct pop
{
    uint8_t opcode = DSP_OPCODE::POP;
};

struct ldarg
{
    uint8_t opcode = DSP_OPCODE::LDARG;
    uint8_t index;
    ldarg(uint8_t index)
        : index(index) { }
};

struct ldarga
{
    uint8_t opcode = DSP_OPCODE::LDARGA;
    uint8_t index;
    uint8_t type;
    ldarga(uint8_t index, datatype_t type)
        : index(index), type(type) { }
};

struct str_i4
{
    uint8_t opcode = DSP_OPCODE::STR_I4;
    uint8_t reg;

    str_i4(uint8_t reg)
        : reg(reg) { }
};

struct neg_
{
    uint8_t opcode = DSP_OPCODE::NEG;
};

struct not_
{
    uint8_t opcode = DSP_OPCODE::NOT;
};

struct add_
{
    uint8_t opcode = DSP_OPCODE::ADD;
};

struct sub_
{
    uint8_t opcode = DSP_OPCODE::SUB;
};

struct mul_
{
    uint8_t opcode = DSP_OPCODE::MUL;
};

struct div_
{
    uint8_t opcode = DSP_OPCODE::DIV;
};

struct div_u
{
    uint8_t opcode = DSP_OPCODE::DIV_U;
};

struct rem_
{
    uint8_t opcode = DSP_OPCODE::REM;
};

struct rem_u
{
    uint8_t opcode = DSP_OPCODE::REM_U;
};

struct clt
{
    uint8_t opcode = DSP_OPCODE::CLT;
};

struct clt_u
{
    uint8_t opcode = DSP_OPCODE::CLT_U;
};

struct cle
{
    uint8_t opcode = DSP_OPCODE::CLE;
};

struct cle_u
{
    uint8_t opcode = DSP_OPCODE::CLE_U;
};

struct ceq
{
    uint8_t opcode = DSP_OPCODE::CEQ;
};

struct cge
{
    uint8_t opcode = DSP_OPCODE::CGE;
};

struct cge_u
{
    uint8_t opcode = DSP_OPCODE::CGE_U;
};

struct cgt
{
    uint8_t opcode = DSP_OPCODE::CGT;
};

struct cgt_u
{
    uint8_t opcode = DSP_OPCODE::CGT_U;
};

struct cne
{
    uint8_t opcode = DSP_OPCODE::CNE;
};

struct conv_i1
{
    uint8_t opcode = DSP_OPCODE::CONV_I1;
};

struct conv_i4
{
    uint8_t opcode = DSP_OPCODE::CONV_I4;
};

struct conv_u1
{
    uint8_t opcode = DSP_OPCODE::CONV_U1;
};

struct conv_u4
{
    uint8_t opcode = DSP_OPCODE::CONV_U4;
};

struct conv_br2
{
    uint8_t opcode = DSP_OPCODE::CONV_BR2;
};

struct conv_r4
{
    uint8_t opcode = DSP_OPCODE::CONV_R4;
};

struct br
{
    int32_t opcode : 8;
    int32_t offset : 24;
    br()
        : opcode((int8_t)DSP_OPCODE::BR) { }
};

struct br_true
{
    int32_t opcode : 8;
    int32_t offset : 24;
    br_true()
        : opcode((int8_t)DSP_OPCODE::BR_TRUE) { }
};

struct br_false
{
    int32_t opcode : 8;
    int32_t offset : 24;
    br_false()
        : opcode((int8_t)DSP_OPCODE::BR_FALSE) { }
};

struct ret
{
    uint8_t opcode = DSP_OPCODE::RET;
};

struct call
{
    int32_t opcode : 8;
    int32_t offset : 24;
    int32_t args_count : 8;
    call()
        : opcode((int8_t)DSP_OPCODE::CALL) { }
};

struct throw_
{
    uint8_t opcode = DSP_OPCODE::THROW;
};

struct pad_t
{
    uint8_t opcode = DSP_OPCODE::PAD_T;
    uint8_t type;
    runtime_shape_t src_shape;
    runtime_shape_t dest_shape;
    padding_config_dim_t dim0;
    padding_config_dim_t dim1;
    padding_config_dim_t dim2;
    padding_config_dim_t dim3;

    pad_t(datatype_t type, runtime_shape_t src_shape, runtime_shape_t dest_shape,
        padding_config_dim_t dim0, padding_config_dim_t dim1,
        padding_config_dim_t dim2, padding_config_dim_t dim3)
        : type(type), src_shape(src_shape), dest_shape(dest_shape), dim0(dim0), dim1(dim1), dim2(dim2), dim3(dim3) { }
};

struct sort_asc_t
{
    uint8_t opcode = DSP_OPCODE::SORT_ASC_T;
    uint8_t type;
    runtime_shape_t src_shape;
    runtime_shape_t dest_shape;
    uint8_t dim;

    sort_asc_t(datatype_t type, runtime_shape_t src_shape,
        runtime_shape_t dest_shape, uint8_t dim)
        : type(type), src_shape(src_shape), dest_shape(dest_shape), dim(dim) { }
};

struct sort_desc_t
{
    uint8_t opcode = DSP_OPCODE::SORT_DESC_T;
    uint8_t type;
    runtime_shape_t src_shape;
    runtime_shape_t dest_shape;
    uint8_t dim;

    sort_desc_t(datatype_t type, runtime_shape_t src_shape,
        runtime_shape_t dest_shape, uint8_t dim)
        : type(type), src_shape(src_shape), dest_shape(dest_shape), dim(dim) { }
};

struct transpose_t
{
    uint8_t opcode = DSP_OPCODE::TRANSPOSE_T;
    uint8_t type;
    runtime_shape_t src_shape;
    runtime_shape_t dest_shape;
    perm_t perm;

    transpose_t(datatype_t type, runtime_shape_t src_shape,
        runtime_shape_t dest_shape, perm_t perm)
        : type(type), src_shape(src_shape), dest_shape(dest_shape), perm(perm) { }
};

struct slice_t
{
    uint8_t opcode = DSP_OPCODE::SLICE_T;
    uint8_t type;
    runtime_shape_t src_shape;
    runtime_shape_t dest_shape;
    slice_config_dim_t dim0;
    slice_config_dim_t dim1;
    slice_config_dim_t dim2;
    slice_config_dim_t dim3;

    slice_t(datatype_t type, runtime_shape_t src_shape,
        runtime_shape_t dest_shape, slice_config_dim_t dim0,
        slice_config_dim_t dim1, slice_config_dim_t dim2,
        slice_config_dim_t dim3)
        : type(type), src_shape(src_shape), dest_shape(dest_shape), dim0(dim0), dim1(dim1), dim2(dim2), dim3(dim3) { }
};

struct convert_t
{
    uint8_t opcode = DSP_OPCODE::CONVERT_T;
    uint8_t src_type;
    runtime_shape_t src_shape;
    uint8_t dest_type;
    runtime_shape_t dest_shape;

    convert_t(datatype_t src_type, runtime_shape_t src_shape,
        datatype_t dest_type, runtime_shape_t dest_shape)
        : src_type(src_type), src_shape(src_shape), dest_type(dest_type), dest_shape(dest_shape) { }
};

struct broadcast_t
{
    uint8_t opcode = DSP_OPCODE::BROADCAST_T;
    uint8_t type;
    runtime_shape_t src_shape;
    runtime_shape_t dest_shape;

    broadcast_t(datatype_t type, runtime_shape_t src_shape,
        runtime_shape_t dest_shape)
        : type(type), src_shape(src_shape), dest_shape(dest_shape) { }
};

struct quantize_t
{
    uint8_t opcode = DSP_OPCODE::QUANTIZE_T;
    uint8_t src_type;
    runtime_shape_t src_shape;
    uint8_t dest_type;
    runtime_shape_t dest_shape;

    quantize_t(datatype_t src_type, runtime_shape_t src_shape,
        datatype_t dest_type, runtime_shape_t dest_shape)
        : src_type(src_type), src_shape(src_shape), dest_type(dest_type), dest_shape(dest_shape) { }
};

struct dequantize_t
{
    uint8_t opcode = DSP_OPCODE::DEQUANTIZE_T;
    uint8_t src_type;
    runtime_shape_t src_shape;
    uint8_t dest_type;
    runtime_shape_t dest_shape;

    dequantize_t(datatype_t src_type, runtime_shape_t src_shape,
        datatype_t dest_type, runtime_shape_t dest_shape)
        : src_type(src_type), src_shape(src_shape), dest_type(dest_type), dest_shape(dest_shape) { }
};

struct clamp_t
{
    uint8_t opcode = DSP_OPCODE::CLAMP_T;
    uint8_t src_type;
    runtime_shape_t src_shape;
    uint8_t dest_type;
    runtime_shape_t dest_shape;

    clamp_t(datatype_t src_type, runtime_shape_t src_shape,
        datatype_t dest_type, runtime_shape_t dest_shape)
        : src_type(src_type), src_shape(src_shape), dest_type(dest_type), dest_shape(dest_shape) { }
};

#pragma pack(pop)

template <DSP_OPCODE Op>
struct dsp_inst;

#define DEFINE_OPCODE(name, value, id) \
    template <>                        \
    struct dsp_inst<name>              \
    {                                  \
        using type = id;               \
    };
#include "gnne_dsp_opcode.def"
#undef DEFINE_OPCODE

template <DSP_OPCODE Op>
using dsp_inst_t = typename dsp_inst<Op>::type;
}
