// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.Passes.Analysis;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Mutators;

public sealed class RemoveBoxingCloner : ExprCloner<Unit>
{
    public RemoveBoxingCloner()
    {
        CloneUnmutated = false;
    }

    protected override BaseExpr VisitLeafCall(Call expr, Unit context)
    {
        if (expr.Target is Boxing boxing)
        {
            var input = (Expr)Visit(expr[Boxing.Input], context);
            if (boxing.NewType is DistributedType dt && dt.TensorType != input.CheckedType)
            {
                // Reshape
                return IR.F.Tensors.Reshape(input, dt.TensorType.Shape);
            }
            else
            {
                return input;
            }
        }

        return base.VisitLeafCall(expr, context);
    }

    protected override BaseExpr VisitLeafTensorConst(TensorConst expr, Unit context)
    {
        return expr.ValueType is DistributedType ? new TensorConst(expr.Value) : base.VisitLeafTensorConst(expr, context);
    }
}
