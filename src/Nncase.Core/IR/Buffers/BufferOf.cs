// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.Buffers;

/// <summary>
/// get the buffer from the input.
/// </summary>
public sealed class BufferOf : Expr
{
    public BufferOf(Expr input)
        : base(new[] { input })
    {
    }

    public Expr Input => Operands[0];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitBufferOf(this, context);
}
