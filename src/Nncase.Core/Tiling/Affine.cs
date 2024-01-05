// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Tiling;

public static class Affine
{
    public static AffineDimExpr Dim(int position) => new AffineDimExpr(position);

    public static AffineDimExpr[] Dims(int count) => Enumerable.Range(0, count).Select(Dim).ToArray();

    public static AffineSymbolExpr Symbol(string name) => new AffineSymbolExpr(name);

    public static AffineSymbolExpr[] Symbols(int count) => Enumerable.Range(0, count).Select(x => Symbol($"s{x}")).ToArray();

    public static AffineDivBinaryExpr FloorDiv(this AffineExpr lhs, AffineConstantExpr rhs) =>
        new AffineDivBinaryExpr(AffineDivBinaryOp.FloorDiv, lhs, rhs);

    public static AffineDivBinaryExpr FloorDiv(this AffineExpr lhs, AffineSymbolExpr rhs) =>
        new AffineDivBinaryExpr(AffineDivBinaryOp.FloorDiv, lhs, rhs);

    public static string ToString(AffineDivBinaryOp binaryOp) => binaryOp switch
    {
        AffineDivBinaryOp.FloorDiv => "floordiv",
        AffineDivBinaryOp.CeilDiv => "ceildiv",
        AffineDivBinaryOp.Mod => "%",
        _ => throw new ArgumentOutOfRangeException(nameof(binaryOp)),
    };
}
