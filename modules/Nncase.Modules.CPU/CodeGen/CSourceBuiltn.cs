// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.CPU;

public static class CSourceBuiltn
{
    public const string BufferType = "buffer_t";

    public const string BufferStruct = @"typedef struct buffer {
    void *vaddr;
    size_t paddr;
    uint32_t *shape;
    uint32_t *stride;
    uint32_t rank;
} buffer_t;";

    public const string MethodTable = @"typedef struct nncase_method_table {
    // float unary
    float (*float_unary_abs)(float);
    float (*float_unary_acos)(float);
    float (*float_unary_acosh)(float);
    float (*float_unary_asin)(float);
    float (*float_unary_asinh)(float);
    float (*float_unary_ceil)(float);
    float (*float_unary_cos)(float);
    float (*float_unary_cosh)(float);
    float (*float_unary_exp)(float);
    float (*float_unary_floor)(float);
    float (*float_unary_log)(float);
    float (*float_unary_logical_not)(float);
    float (*float_unary_neg)(float);
    float (*float_unary_round)(float);
    float (*float_unary_rsqrt)(float);
    float (*float_unary_sign)(float);
    float (*float_unary_sin)(float);
    float (*float_unary_sinh)(float);
    float (*float_unary_sqrt)(float);
    float (*float_unary_square)(float);
    float (*float_unary_tanh)(float);
    // float bianry
    float (*float_binary_add)(float, float);
    float (*float_binary_sub)(float, float);
    float (*float_binary_mul)(float, float);
    float (*float_binary_div)(float, float);
    float (*float_binary_min)(float, float);
    float (*float_binary_max)(float, float);
    float (*float_binary_pow)(float, float);
    float (*float_binary_logical_and)(float, float);
    float (*float_binary_mod)(float, float);
    // int32 bianry
    int32_t (*int32_binary_add)(int32_t, int32_t);
    int32_t (*int32_binary_sub)(int32_t, int32_t);
    int32_t (*int32_binary_mul)(int32_t, int32_t);
    int32_t (*int32_binary_div)(int32_t, int32_t);
    int32_t (*int32_binary_min)(int32_t, int32_t);
    int32_t (*int32_binary_max)(int32_t, int32_t);
    int32_t (*int32_binary_pow)(int32_t, int32_t);
    int32_t (*int32_binary_logical_and)(int32_t, int32_t);
    int32_t (*int32_binary_mod)(int32_t, int32_t);
    // int64 bianry
    int64_t (*int64_binary_add)(int64_t, int64_t);
    int64_t (*int64_binary_sub)(int64_t, int64_t);
    int64_t (*int64_binary_mul)(int64_t, int64_t);
    int64_t (*int64_binary_div)(int64_t, int64_t);
    int64_t (*int64_binary_min)(int64_t, int64_t);
    int64_t (*int64_binary_max)(int64_t, int64_t);
    int64_t (*int64_binary_pow)(int64_t, int64_t);
    int64_t (*int64_binary_logical_and)(int64_t, int64_t);
    int64_t (*int64_binary_mod)(int64_t, int64_t);
    // bool binary
    bool (*bool_binary_and)(bool, bool);
    bool (*bool_binary_or)(bool, bool);
    bool (*bool_binary_xor)(bool, bool);
} nncase_mt_t;";

    public const string Include = @"#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
";

    public const string FixedParameters = "nncase_mt_t* nncase_mt, void* data, void* rdata";

    public const string MainPrologue = $@"void _start(size_t func_id, buffer_t** buffers, {FixedParameters}) {{";

    public const string MainEpilogue = @"}";

    public static string Header => $@"
{Include}

{MethodTable}

{BufferStruct}
";
}
