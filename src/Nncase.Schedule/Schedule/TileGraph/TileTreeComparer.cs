// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using CommunityToolkit.HighPerformance.Helpers;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

public sealed class ITreeNodeComparer : IEqualityComparer<ITreeNode>
{
    public bool Equals(ITreeNode? x, ITreeNode? y) => (x, y) switch
    {
        (null, null) => true,
        (ITreeNode, null) => false,
        (null, ITreeNode) => false,
        (TileNode a, TileNode b) => Compare(a, b),
        (OpNode a, OpNode b) => Compare(a, b),
        _ => throw new System.Diagnostics.UnreachableException(),
    };

    public int GetHashCode([DisallowNull] ITreeNode obj)
    {
        return obj switch
        {
            TileNode t => GetHashCode(t),
            OpNode o => GetHashCode(o),
            _ => throw new System.Diagnostics.UnreachableException(),
        };
    }

    private int GetHashCode([DisallowNull] TileNode a)
    {
        HashCode hascode = default;
        hascode.Add(a.Level);
        hascode.Add(a.DomainRelation.Map);
        foreach (var item in a.Children)
        {
            hascode.Add(item, this);
        }

        return hascode.ToHashCode();
    }

    private int GetHashCode([DisallowNull] OpNode a)
    {
        HashCode code = default;
        code.Add(a.Level);
        code.Add(a.DomainRelation.Map);
        code.Add(StructuralComparisons.StructuralEqualityComparer.GetHashCode(a.DomainBounds));
        code.Add(StructuralComparisons.StructuralEqualityComparer.GetHashCode(a.BufferShapes));
        code.Add(GridHashCode(a.Grid));
        return code.ToHashCode();
    }

    private bool Compare(OpNode a, OpNode b)
    {
        return a.Wrapped.Level.Equals(b.Wrapped.Level) &&
        a.Wrapped.DomainRelation.Map.Equals(b.Wrapped.DomainRelation.Map) &&
        StructuralComparisons.StructuralEqualityComparer.Equals(a.Wrapped.DomainBounds, b.Wrapped.DomainBounds) &&
        StructuralComparisons.StructuralEqualityComparer.Equals(a.Wrapped.BufferShapes, b.Wrapped.BufferShapes) &&
        GridEquals(a.Wrapped.Grid, b.Wrapped.Grid);
    }

    private bool Compare(TileNode a, TileNode b)
    {
        return a.Children.Length.Equals(b.Children.Length) &&
            a.Level.Equals(b.Level) &&
            a.DomainRelation.Map.Equals(b.DomainRelation.Map) &&
            Enumerable.Range(0, a.Children.Length).All(i => Equals(a.Children[i], b.Children[i]));
    }

    private bool GridEquals(Grid x, Grid y)
    {
        return VarTypeEqualityComparer.Instance.Equals(x.DomainParameter, y.DomainParameter) &&
            x.BodyParameters.SequenceEqual(y.BodyParameters, VarTypeEqualityComparer.Instance) &&
            x.AccessMaps.SequenceEqual(y.AccessMaps) &&
            x.Buffers.SequenceEqual(y.Buffers, ExprTypeEqualityComparer.Instance) &&
            x.Reads.SequenceEqual(y.Reads, ExprTypeEqualityComparer.Instance) &&
            new ExprStructuralEqualityVisitor(new[] { (x.DomainParameter, y.DomainParameter) }.Concat(Enumerable.Range(0, x.BodyParameters.Length).Select(i => (x.BodyParameters[i], y.BodyParameters[i])))
            .ToDictionary(p => p.Item1, p => p.Item2)).Visit(x.Body, y.Body);
    }

    private int GridHashCode([DisallowNull] Grid obj)
    {
        return HashCode.Combine(
            obj.DomainParameter.TypeAnnotation,
            HashCode<IRType>.Combine(Enumerable.Range(0, obj.BodyParameters.Length).Select(i => obj.BodyParameters[i].TypeAnnotation).ToArray()),
            HashCode<AffineMap>.Combine(obj.AccessMaps),
            HashCode<IRType>.Combine(Enumerable.Range(0, obj.Buffers.Length).Select(i => obj.Buffers[i].CheckedType).ToArray()),
            HashCode<IRType>.Combine(Enumerable.Range(0, obj.Reads.Length).Select(i => obj.Reads[i].CheckedType).ToArray()),
            new ExprStructuralHashCodeVisitor().Visit(obj.Body));
    }

    private sealed class VarTypeEqualityComparer : IEqualityComparer<Var>
    {
        public static readonly VarTypeEqualityComparer Instance = new VarTypeEqualityComparer();

        public bool Equals(Var? x, Var? y) => x is not null && y is not null && x.TypeAnnotation.Equals(y.TypeAnnotation);

        public int GetHashCode([DisallowNull] Var obj) => obj.TypeAnnotation.GetHashCode();
    }

    private sealed class ExprTypeEqualityComparer : IEqualityComparer<Expr>
    {
        public static readonly ExprTypeEqualityComparer Instance = new ExprTypeEqualityComparer();

        public bool Equals(Expr? x, Expr? y) => x is not null && y is not null && x.CheckedType.Equals(y.CheckedType);

        public int GetHashCode([DisallowNull] Expr obj) => obj.CheckedType.GetHashCode();
    }

    private sealed class ExprStructuralEqualityVisitor : ExprVisitor<bool, Unit, Expr>
    {
        public ExprStructuralEqualityVisitor(Dictionary<Var, Var> varMap)
        {
            VarMap = varMap;
        }

        public Dictionary<Var, Var> VarMap { get; }

        protected override bool DispatchVisit(Expr expr, Expr context)
        {
            if (HasVisited(expr, out var result))
            {
                return result;
            }

            return MarkVisited(expr, VisitExpr(expr, context));
        }

        private bool VisitExpr(Expr expr, Expr other)
        {
            if (expr.GetType() == other.GetType() && expr.Operands.Length == other.Operands.Length)
            {
                if (expr is Var lhs)
                {
                    if (VarMap.TryGetValue(lhs, out var target))
                    {
                        return target.Equals(other);
                    }

                    return other is Var rhs && lhs.TypeAnnotation.Equals(rhs.TypeAnnotation);
                }

                return Enumerable.Range(0, expr.Operands.Length).All(i => Visit(expr.Operands[i], other.Operands[i]));
            }

            return false;
        }
    }

    private sealed class ExprStructuralHashCodeVisitor : ExprVisitor<int, Unit>
    {
        protected override int DispatchVisit(Expr expr)
        {
            if (HasVisited(expr, out var result))
            {
                return result;
            }

            return MarkVisited(expr, VisitExpr(expr));
        }

        private int VisitExpr(Expr expr)
        {
            if (expr.Operands.Length == 0)
            {
                if (expr is Var @var)
                {
                    return @var.TypeAnnotation.GetHashCode();
                }

                return expr.GetHashCode();
            }
            else
            {
                return Enumerable.Range(0, expr.Operands.Length).Select(i => Visit(expr.Operands[i])).Aggregate(expr.GetType().GetHashCode(), HashCode.Combine);
            }
        }
    }
}
