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
/// flatten sequential.
/// </summary>
public sealed class FlattenSequential : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafSequential(Sequential expr)
    {
        if (expr.Fields.AsValueEnumerable().Any(x => x is Sequential))
        {
            return Sequential.Flatten(expr.Fields);
        }

        return expr;
    }
}
