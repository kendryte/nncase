// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.CPU;

public static class CSourceBuiltn
{

    public const string BufferType = "buffer_t";

    public const string BufferStruct = @"typedef struct buffer {
    void *ptr;
    int *shape;
    int *stride;
    int rank;
} buffer_t;";

    public const string Include = @"#include<stdio.h>";

    public static string Header => Include + "\n" + BufferStruct;

}