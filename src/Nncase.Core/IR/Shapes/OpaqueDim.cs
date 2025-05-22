// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

public abstract class OpaqueDim : Dimension
{
    protected OpaqueDim(BaseExpr[] operands)
        : base(operands)
    {
    }
}
