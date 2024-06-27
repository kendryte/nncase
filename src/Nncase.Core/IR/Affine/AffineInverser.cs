// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.Utilities;

namespace Nncase.IR.Affine;

internal sealed class AffineInverseCollector : AffineExprVisitor<bool, Unit>
{
    public AffineInverseCollector()
    {
        DomainNums = 0;
        ExtentNums = 0;
        Valid = false;
    }

    public bool Valid { get; private set; }

    public int DomainNums { get; private set; }

    public int ExtentNums { get; private set; }

    protected override bool VisitLeafAffineDim(AffineDim expr, Unit context)
    {
        DomainNums++;
        Valid |= DomainNums == 1;
        return true;
    }

    protected override bool VisitLeafAffineExtent(AffineExtent expr, Unit context)
    {
        ExtentNums++;
        Valid |= ExtentNums == 1;
        return true;
    }

    protected override bool VisitLeafAffineSymbol(AffineSymbol expr, Unit context)
    {
        return false;
    }

    protected override bool VisitLeafAffineConstant(AffineConstant expr, Unit context)
    {
        return false;
    }

    protected override bool VisitLeafAffineAddBinary(AffineAddBinary expr, Unit context)
    {
        var lhs = Visit(expr.Lhs, context);
        var rhs = Visit(expr.Rhs, context);
        Valid &= CheckBinary(lhs, rhs);
        return lhs | rhs;
    }

    protected override bool VisitLeafAffineDivBinary(AffineDivBinary expr, Unit context)
    {
        var lhs = Visit(expr.Lhs, context);
        var rhs = Visit(expr.Rhs, context);
        Valid &= CheckBinary(lhs, rhs);
        return lhs | rhs;
    }

    protected override bool VisitLeafAffineMulBinary(AffineMulBinary expr, Unit context)
    {
        var lhs = Visit(expr.Lhs, context);
        var rhs = Visit(expr.Rhs, context);
        Valid &= CheckBinary(lhs, rhs);
        return lhs | rhs;
    }

    private bool CheckBinary(bool lhs, bool rhs) => (lhs, rhs) switch
    {
        (true, true) => false,
        _ => true,
    };
}

internal sealed class AffineInverser<T> : AffineExprVisitor<AffineExpr, AffineExpr>
    where T : AffineExpr
{
    private readonly Dictionary<AffineExpr, bool> _symMemo;

    public AffineInverser(Dictionary<AffineExpr, bool> symMemo)
    {
        _symMemo = symMemo;
    }

    public T? IndependentVariable { get; private set; }

    protected internal override AffineExpr VisitAffineConstant(AffineConstant expr, AffineExpr target) => expr;

    protected internal override AffineExpr VisitAffineDim(AffineDim expr, AffineExpr target)
    {
        if (IndependentVariable is null && typeof(T) == typeof(AffineDim))
        {
            IndependentVariable = (T)(object)expr;
        }

        return target;
    }

    protected internal override AffineExpr VisitAffineExtent(AffineExtent expr, AffineExpr target)
    {
        if (IndependentVariable is null && typeof(T) == typeof(AffineExtent))
        {
            IndependentVariable = (T)(object)expr;
        }

        return target;
    }

    protected internal override AffineExpr VisitAffineSymbol(AffineSymbol expr, AffineExpr target)
    {
        return expr;
    }

    protected internal override AffineExpr VisitAffineAddBinary(AffineAddBinary expr, AffineExpr target)
    {
        // {((2 * d1) + 3)}
        return ConstFolding((_symMemo[expr.Lhs], _symMemo[expr.Rhs]) switch
        {
            (true, false) => Visit(expr.Lhs, new AffineAddBinary(target, ConstFolding(-Visit(expr.Rhs, null!)))),
            (false, true) => Visit(expr.Rhs, new AffineAddBinary(target, ConstFolding(-Visit(expr.Lhs, null!)))),
            (false, false) => expr,
            _ => throw new System.Diagnostics.UnreachableException(),
        });
    }

    protected internal override AffineExpr VisitAffineMulBinary(AffineMulBinary expr, AffineExpr target)
    {
        return ConstFolding((_symMemo[expr.Lhs], _symMemo[expr.Rhs]) switch
        {
            (false, true) => Visit(expr.Rhs, ConstFolding(new AffineDivBinary(AffineDivBinaryOp.CeilDiv, target, (AffineSymbolBase)Visit(expr.Lhs, null!)))),
            (false, false) => expr,
            _ => throw new System.Diagnostics.UnreachableException(),
        });
    }

    protected internal override AffineExpr VisitAffineDivBinary(AffineDivBinary expr, AffineExpr target)
    {
        return ConstFolding((_symMemo[expr.Lhs], _symMemo[expr.Rhs]) switch
        {
            (true, false) => Visit(expr.Lhs, ConstFolding(new AffineMulBinary((AffineSymbolBase)Visit(expr.Rhs, null!), target))),
            (false, false) => expr,
            _ => throw new System.Diagnostics.UnreachableException(),
        });
    }

    private AffineExpr ConstFolding(AffineExpr expr) => expr switch
    {
        AffineAddBinary e => (e.Lhs, e.Rhs) switch
        {
            (AffineConstant lhs, AffineConstant rhs) => lhs.Value + rhs.Value,
            _ => e,
        },
        AffineMulBinary e => (e.Lhs, e.Rhs) switch
        {
            (AffineConstant lhs, AffineConstant rhs) => lhs.Value * rhs.Value,
            _ => e,
        },
        AffineDivBinary e => (e.Lhs, e.Rhs) switch
        {
            (AffineConstant lhs, AffineConstant rhs) => e.BinaryOp switch
            {
                AffineDivBinaryOp.FloorDiv => lhs.Value / rhs.Value,
                AffineDivBinaryOp.CeilDiv => (lhs.Value + rhs.Value - 1) / rhs.Value,
                AffineDivBinaryOp.Mod => lhs.Value % rhs.Value,
                _ => e,
            },
            _ => e,
        },
        _ => expr,
    };
}
