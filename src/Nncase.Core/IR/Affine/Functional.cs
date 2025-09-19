// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Affine;
using Nncase.IR.Affine.Builders;

namespace Nncase.IR.F;

public static class Affine
{
    public static AffineDim Dim(int position) => new AffineDim(position);

    public static AffineDim[] Dims(int count) => Enumerable.Range(0, count).Select(Dim).ToArray();

    public static AffineExtent Extent(int position) => new AffineExtent(position);

    public static AffineDomain Domain(int position) => new AffineDomain(Dim(position), Extent(position));

    public static AffineDomain[] Domains(int count) => Enumerable.Range(0, count).Select(Domain).ToArray();

    public static AffineSymbol Symbol(int position) => new AffineSymbol(position);

    public static AffineSymbol[] Symbols(int count) => Enumerable.Range(0, count).Select(Symbol).ToArray();

    public static AffineDivBinary FloorDiv(this AffineExpr lhs, AffineConstant rhs) =>
        new AffineDivBinary(AffineDivBinaryOp.FloorDiv, lhs, rhs);

    public static AffineDivBinary FloorDiv(this AffineExpr lhs, AffineSymbol rhs) =>
        new AffineDivBinary(AffineDivBinaryOp.FloorDiv, lhs, rhs);

    public static string ToString(AffineDivBinaryOp binaryOp, AffineExpr lhs, AffineExpr rhs) => binaryOp switch
    {
        AffineDivBinaryOp.FloorDiv => $"floor({lhs} / {rhs})",
        AffineDivBinaryOp.CeilDiv => $"ceil({lhs} / {rhs})",
        AffineDivBinaryOp.Mod => $"({lhs} % {rhs})",
        _ => throw new ArgumentOutOfRangeException(nameof(binaryOp)),
    };

    public static string GetDisplayString(AffineDivBinaryOp binaryOp, AffineExpr lhs, AffineExpr rhs, ReadOnlySpan<AffineSymbol> symbols) => binaryOp switch
    {
        AffineDivBinaryOp.FloorDiv => $"floor({lhs.GetDisplayString(symbols)} / {rhs.GetDisplayString(symbols)})",
        AffineDivBinaryOp.CeilDiv => $"ceil({lhs.GetDisplayString(symbols)} / {rhs.GetDisplayString(symbols)})",
        AffineDivBinaryOp.Mod => $"({lhs.GetDisplayString(symbols)} % {rhs.GetDisplayString(symbols)})",
        _ => throw new ArgumentOutOfRangeException(nameof(binaryOp)),
    };

    public static Load Load(Expr source, AffineMap region) => new Load(source, region);

    public static For For(int memoryLevel, AffineMap domain, Expr body) => new For(memoryLevel, domain, body);

    public static IGridBuilder Grid() => new GridBuilder();
}
