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
    public static IntegerSetLibrary.basic_map AsIslMap(this AffineMap map, string domainPrefix, IEnumerable<int> domainBounds)
    {
        var domain = map.Domains.AsValueEnumerable().Select(dim => dim.Offset.ToString()).ToArray();
        var cvt = new AffineExprToStringConverter(domain);
        var range = new string[map.Results.Length];
        for (int i = 0; i < map.Results.Length; i++)
        {
            range[i] = cvt.Visit(map.Results[i].Offset);
        }

        var constrains = domain.Zip(domainBounds).Select(p => $" 0 <= {p.First} < {p.Second}").ToArray();

        return new(IntegerSetLibrary.ctx.Instance, $"{{ {domainPrefix}[{string.Join(',', domain)}] -> [{string.Join(',', range)}] : {string.Join(" and ", constrains)} }}");
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
