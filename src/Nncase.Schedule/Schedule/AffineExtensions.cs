// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Utilities;

namespace Nncase.Schedule;

internal static class AffineExtensions
{
    public static AffineRelation AsRelation(this AffineMap map)
    {
        return new AffineRelation(map.Domains.AsValueEnumerable().Select(d => d.Offset).ToArray(), map.Symbols, map.Results.AsValueEnumerable().Select(i => i.Offset).ToArray());
    }
}

internal sealed class AffineExprToStringConverter : ExprVisitor<string, Unit>
{
    public AffineExprToStringConverter(params string[] dims)
    {
        Dims = new();
        if (dims.Any())
        {
            for (int i = 0; i < dims.Length; i++)
            {
                Dims[i] = dims[i];
            }
        }
    }

    public Dictionary<int, string> Dims { get; }

    protected override string VisitLeafAffineDim(AffineDim expr)
    {
        if (!Dims.TryGetValue(expr.Position, out var v))
        {
            v = $"d{expr.Position}";
            Dims.Add(expr.Position, v);
        }

        return v;
    }

    protected override string VisitLeafAffineConstant(AffineConstant expr) => expr.Value.ToString();

    protected override string VisitLeafAffineAddBinary(AffineAddBinary expr) =>
        $"({ExprMemo[expr.Lhs]} + {ExprMemo[expr.Rhs]})";

    protected override string VisitLeafAffineMulBinary(AffineMulBinary expr) => $"({ExprMemo[expr.Lhs]} * {ExprMemo[expr.Rhs]})";

    protected override string VisitLeafAffineDivBinary(AffineDivBinary expr) => expr.BinaryOp switch
    {
        AffineDivBinaryOp.FloorDiv => $"({ExprMemo[expr.Lhs]} / {ExprMemo[expr.Rhs]})",
        AffineDivBinaryOp.CeilDiv => $"({ExprMemo[expr.Lhs]} // {ExprMemo[expr.Rhs]})",
        AffineDivBinaryOp.Mod => $"({ExprMemo[expr.Lhs]} mod {ExprMemo[expr.Rhs]})",
        _ => throw new NotSupportedException(expr.BinaryOp.ToString()),
    };
}
