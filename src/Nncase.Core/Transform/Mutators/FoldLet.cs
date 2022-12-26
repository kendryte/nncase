// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform.Mutators;

/// <summary>
/// unroll loop.
/// </summary>
internal sealed class FoldLet : ExprMutator
{
    /// <inheritdoc/>
    public override Expr MutateLeaf(TIR.Let expr)
    {
        if (expr.Expression is Const @const)
        {
            return new Substitutor(e =>
              {
                  if (object.ReferenceEquals(e, expr.Var))
                  {
                      return @const;
                  }

                  return null;
              }).Visit(expr.Body);
        }

        return expr;
    }
}
