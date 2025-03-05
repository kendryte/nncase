// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Shape inference.
/// </summary>
public sealed class InferRangePass : FunctionPass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="InferRangePass"/> class.
    /// </summary>
    public InferRangePass()
    {
    }

    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction pre, RunPassContext options)
    {
        var visitor = new InferRangeVisitor();
        visitor.Visit(pre);
        return Task.FromResult(pre);
    }
}

internal sealed class InferRangeVisitor : ExprVisitor<ValueRange<double>, Unit>
{
    public InferRangeVisitor()
        : base(visitAttributes: true)
    {
    }

    public override Unit DefaultVisitTypeLeaf(IRType type, Unit context) => default;

    protected override ValueRange<double> DispatchVisit(Expr expr)
    {
        if (expr.Metadata.Range is null)
        {
            expr.Metadata.Range = base.DispatchVisit(expr);
        }

        return expr.Metadata.Range!.Value;
    }

    protected override ValueRange<double> DefaultVisitLeaf(Expr expr)
    {
        return expr.Metadata.Range ?? ValueRange<double>.Full;
    }

    /// <inheritdoc/>
    protected override ValueRange<double> VisitLeafCall(Call expr)
    {
        var range = expr.Target switch
        {
            Op op => InferenceOp(op, expr),
            BaseFunction => ValueRange<double>.Full,
            _ => ValueRange<double>.Full,
        };
        return range;
    }

    protected override ValueRange<double> VisitLeafTensorConst(TensorConst expr)
    {
        // QuantParam
        if (expr.Value.ElementType is PrimType)
        {
            var value = expr.Value.ToArray<double>();
            return value.Length == 0 ? new ValueRange<double>(0, 0) : new ValueRange<double>(value.Min(), value.Max());
        }
        else
        {
            return ValueRange<double>.Full;
        }
    }

    protected override ValueRange<double> VisitLeafShape(Shape expr)
    {
        if (!expr.Any())
        {
            return ValueRange<double>.Full;
        }

        var ranges = expr.Select(x => x.IsFixed ? new ValueRange<double>(x.FixedValue, x.FixedValue) : Visit(x.Value)).ToArray();
        return new ValueRange<double>(ranges.Min(x => x.Min), ranges.Max(x => x.Max));
    }

    protected override ValueRange<double> VisitLeafTuple(IR.Tuple expr)
    {
        var ranges = expr.Fields.AsValueEnumerable().Select(Visit).ToArray();
        return new ValueRange<double>(ranges.Min(x => x.Min), ranges.Max(x => x.Max));
    }

    private ValueRange<double> InferenceOp(Op op, Call expr)
    {
        return op switch
        {
            Reshape => Visit(expr[Reshape.Input]),
            Slice => Visit(expr[Slice.Input]),
            Gather => Visit(expr[Gather.Input]),
            GetItem => Visit(expr[GetItem.Input]),
            Concat => Visit(expr[Concat.Input]),
            Binary binary => InferenceBinary(expr, binary.BinaryOp),
            Squeeze => Visit(expr[Squeeze.Input]),
            Unsqueeze => Visit(expr[Unsqueeze.Input]),
            Stack => Visit(expr[Stack.Inputs]),
            Select => InferenceSelect(expr),
            _ => ValueRange<double>.Full,
        };
    }

    private ValueRange<double> InferenceSelect(Call expr)
    {
        var then = Visit(expr[Select.TrueValue]);
        var @else = Visit(expr[Select.FalseValue]);
        return new(Math.Min(then.Min, @else.Min), Math.Max(then.Max, @else.Max));
    }

    private ValueRange<double> InferenceBinary(Call expr, BinaryOp op)
    {
        var lhs = Visit(expr[Binary.Lhs]);
        var rhs = Visit(expr[Binary.Rhs]);

        return op switch
        {
            BinaryOp.Add => new(lhs.Min + rhs.Min, lhs.Max + rhs.Max),
            BinaryOp.Sub => new(lhs.Min - rhs.Max, lhs.Max - rhs.Min),
            BinaryOp.Mul => VisitMul(lhs, rhs),
            BinaryOp.Div => VisitDiv(lhs, rhs),
            BinaryOp.Max => new(Math.Max(lhs.Min, rhs.Min), Math.Max(lhs.Max, rhs.Max)),
            BinaryOp.Min => new(Math.Min(lhs.Min, rhs.Min), Math.Min(lhs.Max, rhs.Max)),
            _ => ValueRange<double>.Full,
        };
    }

    private ValueRange<double> VisitDiv(ValueRange<double> lhs, ValueRange<double> rhs)
    {
        if (rhs.Min <= 0 && rhs.Max >= 0)
        {
            return ValueRange<double>.Full;
        }

        var values = new[]
        {
            lhs.Min / rhs.Min,
            lhs.Min / rhs.Max,
            lhs.Max / rhs.Min,
            lhs.Max / rhs.Max,
        };
        return new ValueRange<double>(values.Min(), values.Max());
    }

    private ValueRange<double> VisitMul(ValueRange<double> lhs, ValueRange<double> rhs)
    {
        var values = new[]
        {
            lhs.Min * rhs.Min,
            lhs.Min * rhs.Max,
            lhs.Max * rhs.Min,
            lhs.Max * rhs.Max,
        };
        return new ValueRange<double>(values.Min(), values.Max());
    }
}
