// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if false
/// <summary>
/// Fold IR avoid too much calls/const
/// </summary>
internal sealed class FoldTIR : TirMutator
{
    private readonly Dictionary<Expr, Expr> _exprSEqualMemo;

private readonly Dictionary<IMutatable, IMutatable> _mutateableSEqualMemo;

public FoldTIR()
    {
        _exprSEqualMemo = new Dictionary<Expr, Expr>();
        _mutateableSEqualMemo = new(); // NOTE is this struct equal?
    }

// NOTE do not modify follow expression
    public override Expr MutateLeaf(Op expr) => expr;
    public override Expr MutateLeaf(Var expr) => expr;
    public override Expr MutateLeaf(None expr) => expr;
    public override Expr MutateLeaf(TIR.Buffer expr) => expr;

/// <inheritdoc/>
    public override Expr DefaultMutateLeaf(Expr expr)
    {
        if (!_exprSEqualMemo.TryGetValue(expr, out var result))
        {
            result = expr;
            _exprSEqualMemo.Add(expr, result);
        }

return result;
    }

/// <inheritdoc/>
    public override IMutatable DefaultMutateLeaf(IMutatable mutatable)
    {
        if (!_mutateableSEqualMemo.TryGetValue(mutatable, out var result))
        {
            result = mutatable;
            _mutateableSEqualMemo.Add(mutatable, result);
        }
        return result;
    }
}
#endif
