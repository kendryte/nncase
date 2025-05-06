// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Shapes;

namespace Nncase.IR;

/// <summary>
/// Leaf expression equality comparer.
/// </summary>
public sealed class LeafExprEqualityComparer : IEqualityComparer<BaseExpr>
{
    /// <summary>
    /// Gets instance.
    /// </summary>
    public static LeafExprEqualityComparer Instance { get; } = new();

    /// <inheritdoc/>
    public bool Equals(BaseExpr? x, BaseExpr? y)
    {
        if (x == null && y == null)
        {
            return true;
        }

        if (x == null || y == null)
        {
            return false;
        }

        if (x.GetType() != y.GetType())
        {
            return false;
        }

        return (x, y) switch
        {
            (Var tx, Var ty) => tx.Equals(ty),
            (Const tx, Const ty) => tx.Equals(ty),
            (Fusion tx, Fusion ty) => tx.Equals(ty),

            // note think of primfunc/primfunc wrapper as a black box.
            (TIR.PrimFunction tx, TIR.PrimFunction ty) => ReferenceEquals(tx, ty),
            (PrimFunctionWrapper tx, PrimFunctionWrapper ty) => ReferenceEquals(tx, ty),
            (Function tx, Function ty) => tx.Parameters.Length == ty.Parameters.Length,
            (Tuple tx, Tuple ty) => tx.Count == ty.Count,
            (Call tx, Call ty) => tx.Arguments.Length == ty.Arguments.Length,
            (Op tx, Op ty) => tx.Equals(ty),
            (IR.If, IR.If) => true,
            (Marker tx, Marker ty) => tx.Name == ty.Name,
            (None tx, None ty) => tx.Equals(ty),

            // Dimension
            (AsDim, AsDim) => true,
            (UnknownDim, UnknownDim) => true,
            (DimFraction, DimFraction) => true,
            (DimRemainder, DimRemainder) => true,
            (DimAbs, DimAbs) => true,
            (DimClamp, DimClamp) => true,
            (DimCompareAndSelect, DimCompareAndSelect) => true,
            (DimMin, DimMin) => true,
            (DimMax, DimMax) => true,
            (DimPositive, DimPositive) => true,
            (DimAt, DimAt) => true,
            (DimVar tx, DimVar ty) => tx.Equals(ty),
            (DimConst tx, DimConst ty) => tx.Equals(ty),
            (DimPower tx, DimPower ty) => tx.Power.Equals(ty.Power),
            (DimProduct tx, DimProduct ty) => tx.Scale.Equals(ty.Scale) && tx.Operands.Length == ty.Operands.Length,
            (DimSum tx, DimSum ty) => tx.Bias.Equals(ty.Bias) && tx.Operands.Length == ty.Operands.Length,

            // Padding
            (Padding, Padding) => true,
            (Paddings tx, Paddings ty) => tx.Rank == ty.Rank,

            // Shape
            (UnrankedShape, UnrankedShape) => true,
            (InvalidShape, InvalidShape) => true,
            (RankedShape tx, RankedShape ty) => tx.Rank == ty.Rank && tx.Kind == ty.Kind,
            (ShapeVar tx, ShapeVar ty) => tx.Equals(ty),
            _ => throw new InvalidOperationException("Invalid expression type."),
        };
    }

    /// <inheritdoc/>
    public int GetHashCode(BaseExpr obj)
    {
        return obj switch
        {
            Var x => x.GetHashCode(),
            Const x => x.GetHashCode(),
            Function x => ReferenceEqualityComparer.Instance.GetHashCode(x),
            Fusion x => x.GetHashCode(),
            TIR.PrimFunction x => ReferenceEqualityComparer.Instance.GetHashCode(x),
            PrimFunctionWrapper x => ReferenceEqualityComparer.Instance.GetHashCode(x),
            Tuple x => x.Count.GetHashCode(),
            Call x => x.Arguments.Length.GetHashCode(),
            Op x => x.GetHashCode(),
            Marker x => x.Name.GetHashCode(StringComparison.Ordinal),
            None x => x.GetHashCode(),
            IR.If x => x.GetType().GetHashCode(),

            // Dimension
            AsDim or UnknownDim or DimFraction or DimRemainder or DimAbs or DimClamp or DimCompareAndSelect
            or DimMin or DimMax or DimPositive or DimAt => obj.GetType().GetHashCode(),
            DimVar or DimConst => obj.GetHashCode(),
            DimPower x => x.Power.GetHashCode(),
            DimProduct x => HashCode.Combine(x.Scale, x.Operands.Length),
            DimSum x => HashCode.Combine(x.Bias, x.Operands.Length),

            // Padding
            Padding x => x.GetType().GetHashCode(),
            Paddings x => x.Rank.GetHashCode(),

            // Shape
            UnrankedShape or InvalidShape => obj.GetType().GetHashCode(),
            RankedShape x => HashCode.Combine(x.Rank, x.Kind),
            ShapeVar x => x.GetHashCode(),

            _ => throw new InvalidOperationException("Invalid expression type."),
        };
    }
}
