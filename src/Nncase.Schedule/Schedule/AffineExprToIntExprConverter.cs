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
    private readonly Dictionary<int, IntExpr> _dims = new();

    public AffineExprToIntExprConverter(Solver solver, params IntExpr[] dims)
    {
        _solver = solver;
        if (dims.Any())
        {
            for (int i = 0; i < dims.Length; i++)
            {
                _dims[i] = dims[i];
            }
        }
    }

    protected override IntExpr VisitLeafAffineDim(AffineDim expr)
    {
        if (!_dims.TryGetValue(expr.Position, out var v))
        {
            v = _solver.MakeIntVar(1, int.MaxValue, $"d{expr.Position}_v");
            _dims.Add(expr.Position, v);
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
