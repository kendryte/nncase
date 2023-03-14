// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// unroll loop.
/// </summary>
public sealed class FoldConstTuple : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafTuple(IR.Tuple expr)
    {
        if (expr.Fields.AsValueEnumerable().All(x => x is Const))
        {
            return new TupleConst(new TupleValue(expr.Fields.AsValueEnumerable().Select(x => Value.FromConst((Const)x)).ToArray()));
        }

        return expr;
    }
}
