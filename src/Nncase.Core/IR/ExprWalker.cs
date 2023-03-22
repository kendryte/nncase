// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

public abstract class ExprWalker<TContext> : ExprVisitor<Unit, Unit, TContext>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ExprWalker{TContext}"/> class.
    /// </summary>
    /// <param name="visitOtherFunctions">Vist other functions.</param>
    public ExprWalker(bool visitOtherFunctions = false)
        : base(visitOtherFunctions)
    {
    }

    protected override Unit DefaultVisitLeaf(Expr expr, TContext context) => default;
}

public abstract class ExprWalker : ExprVisitor<Unit, Unit>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ExprWalker"/> class.
    /// </summary>
    /// <param name="visitOtherFunctions">Vist other functions.</param>
    public ExprWalker(bool visitOtherFunctions = false)
        : base(visitOtherFunctions)
    {
    }

    protected override Unit DefaultVisitLeaf(Expr expr) => default;
}
