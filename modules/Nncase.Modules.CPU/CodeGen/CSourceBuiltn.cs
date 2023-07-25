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
} nncase_mt_t;";

    public const string Include = @"#include <stdint.h>
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
