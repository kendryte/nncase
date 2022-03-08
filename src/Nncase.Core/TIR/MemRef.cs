// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.TIR;


public sealed record MemRef(IRType ElemType, int DataAlignment = 0) : Expr
{
    public Shape Shape => ElemType switch
    {
        TensorType type => type.Shape,
        _ => throw new NotSupportedException(ElemType.ToString()),
    };

    public DataType DType => ElemType switch
    {
        TensorType type => type.DType,
        _ => throw new NotSupportedException(ElemType.ToString()),
    };

    public Expr Addr => IR.F.Buffer.DDrOf(this);

    public Expr BaseMent => IR.F.Buffer.BaseMentOf(this);
}