using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Transform.Mutators;


/// <summary>
/// Fold IR avoid too much calls/const
/// </summary>
internal sealed class FoldIR : ExprMutator
{
    private readonly Dictionary<Expr, Expr> _exprSEqualMemo;

    public FoldIR()
    {
        _exprSEqualMemo = new Dictionary<Expr, Expr>();
    }

    public override Expr DefaultMutateLeaf(Expr expr)
    {
        if (!_exprSEqualMemo.TryGetValue(expr, out var result))
        {
            result = expr;
            _exprSEqualMemo.Add(expr, result);
        }

        return result;
    }
}
