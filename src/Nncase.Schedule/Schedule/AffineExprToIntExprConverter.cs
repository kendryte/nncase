// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule;

internal sealed class AffineExprToIntExprConverter : ExprVisitor<IntExpr, Unit>
{
    private readonly Solver _solver;
    private readonly Dictionary<int, IntExpr> _extents = new();

    public AffineExprToIntExprConverter(Solver solver, params IntExpr[] extents)
    {
        _solver = solver;
        if (extents.Any())
        {
            for (int i = 0; i < extents.Length; i++)
            {
                _extents[i] = extents[i];
            }
        }
    }

    protected override IntExpr VisitLeafAffineExtent(AffineExtent expr)
    {
        if (!_extents.TryGetValue(expr.Position, out var v))
        {
            v = _solver.MakeIntVar(1, int.MaxValue, $"d{expr.Position}_v");
            _extents.Add(expr.Position, v);
        }

        return v;
    }

    protected override IntExpr VisitLeafAffineConstant(AffineConstant expr) =>
        _solver.MakeIntConst(expr.Value);

    protected override IntExpr VisitLeafAffineAddBinary(AffineAddBinary expr) =>
        ExprMemo[expr.Lhs] + ExprMemo[expr.Rhs];

    protected override IntExpr VisitLeafAffineMulBinary(AffineMulBinary expr) =>
        ExprMemo[expr.Lhs] * ExprMemo[expr.Rhs];

    protected override IntExpr VisitLeafAffineDivBinary(AffineDivBinary expr) =>
        expr.BinaryOp switch
        {
            AffineDivBinaryOp.FloorDiv => _solver.MakeDiv(ExprMemo[expr.Lhs], ExprMemo[expr.Rhs]),
            AffineDivBinaryOp.CeilDiv => ExprMemo[expr.Lhs].CeilDiv(ExprMemo[expr.Rhs]),
            AffineDivBinaryOp.Mod => _solver.MakeModulo(ExprMemo[expr.Lhs], ExprMemo[expr.Rhs]),
            _ => throw new ArgumentOutOfRangeException(expr.BinaryOp.ToString()),
        };
}
