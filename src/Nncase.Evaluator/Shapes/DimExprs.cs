// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using SMath = System.Math;

namespace Nncase.Evaluator;

internal partial class EvaluateVisitor
{
    protected override IValue VisitLeafDimVar(DimVar expr)
    {
        if (!_varsValues.TryGetValue(expr, out var value))
        {
            throw new ArgumentException($"Must Set Input For Var {expr.Name}!");
        }

        if (expr.CheckedType is TensorType { IsScalar: true } tt && tt.DType == DataTypes.Int64)
        {
            return value;
        }

        throw new ArgumentException($"DimVar {expr.Name} must be a scalar int64 tensor!");
    }

    protected override IValue VisitLeafDimAbs(DimAbs expr) => Value.FromTensor(GetDimValue(expr.Operand));

    protected override IValue VisitLeafDimClamp(DimClamp expr) => Value.FromTensor(
        SMath.Clamp(
            GetDimValue(expr.Operand),
            GetDimValue(expr.MinValue),
            GetDimValue(expr.MaxValue)));

    protected override IValue VisitLeafDimCompareAndSelect(DimCompareAndSelect expr) => Value.FromTensor(
        GetDimValue(expr.Value) == GetDimValue(expr.Expected)
            ? GetDimValue(expr.TrueValue)
            : GetDimValue(expr.FalseValue));

    protected override IValue VisitLeafDimConst(DimConst expr) => Value.FromTensor(expr.Value);

    protected override IValue VisitLeafDimFraction(DimFraction expr)
    {
        var numerator = GetDimValue(expr.Numerator);
        var denominator = GetDimValue(expr.Denominator);
        return Value.FromTensor(expr.DivMode switch
        {
            DimDivideMode.FloorDiv => numerator / denominator,
            DimDivideMode.CeilDiv => MathUtility.CeilDiv(numerator, denominator),
            _ => throw new NotSupportedException($"Unsupported DimDivideMode {expr.DivMode}"),
        });
    }

    protected override IValue VisitLeafDimMax(DimMax expr)
    {
        var operands = expr.Operands.AsValueEnumerable().Select(GetDimValue).ToArray();
        return Value.FromTensor(operands.Max());
    }

    protected override IValue VisitLeafDimMin(DimMin expr)
    {
        var operands = expr.Operands.AsValueEnumerable().Select(GetDimValue).ToArray();
        return Value.FromTensor(operands.Min());
    }

    protected override IValue VisitLeafDimPositive(DimPositive expr)
    {
        var operand = GetDimValue(expr.Operand);
        var extent = GetDimValue(expr.Extent);
        return Value.FromTensor(operand >= 0 ? operand : operand + extent);
    }

    protected override IValue VisitLeafDimPower(DimPower expr) => Value.FromTensor(
        (long)SMath.Pow(GetDimValue(expr.Dim), expr.Power));

    protected override IValue VisitLeafDimProduct(DimProduct expr)
    {
        var operands = expr.Operands.AsValueEnumerable().Select(GetDimValue).ToArray();
        return Value.FromTensor(operands.Aggregate(expr.Scale, (x, y) => x * y));
    }

    protected override IValue VisitLeafDimRemainder(DimRemainder expr) => Value.FromTensor(
        GetDimValue(expr.Numerator) % GetDimValue(expr.Denominator));

    protected override IValue VisitLeafDimSum(DimSum expr)
    {
        var operands = expr.Operands.AsValueEnumerable().Select(GetDimValue).ToArray();
        return Value.FromTensor(operands.Aggregate(expr.Bias, (x, y) => x + y));
    }

    protected override IValue VisitLeafAsDim(AsDim expr) => Value.FromTensor(GetDimValue(expr.Dim));

    private long GetDimValue(Expr dimension) => ExprMemo[dimension].AsTensor().ToScalar<long>();
}
