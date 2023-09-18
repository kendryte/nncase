// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Unary"/>.
/// </summary>
public class UnaryEvaluator : IEvaluator<Unary>, ITypeInferencer<Unary>, ICostEvaluator<Unary>, IOpPrinter<Unary>, IShapeEvaluator<Unary>, IMetricEvaluator<Unary>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unary unary)
    {
        var input_tensor = context.GetArgumentValueAsTensor(unary, Unary.Input);
        if (input_tensor.Shape.IsScalar)
        {
            if (input_tensor.ElementType == DataTypes.Int32)
            {
                return Value.FromTensor(Tensor.FromScalar<int>(Compute_int(input_tensor.ToScalar<int>(), unary.UnaryOp)));
            }
            else if (input_tensor.ElementType == DataTypes.Float32)
            {
                return Value.FromTensor(Tensor.FromScalar<float>(Compute_float(input_tensor.ToScalar<float>(), unary.UnaryOp)));
            }
        }

        var input = context.GetOrtArgumentValue(unary, Unary.Input);
        var result = unary.UnaryOp switch
        {
            UnaryOp.Abs => OrtKI.Abs(input),
            UnaryOp.Acos => OrtKI.Acos(input),
            UnaryOp.Acosh => OrtKI.Acosh(input),
            UnaryOp.Asin => OrtKI.Asin(input),
            UnaryOp.Asinh => OrtKI.Asinh(input),
            UnaryOp.Ceil => OrtKI.Ceil(input),
            UnaryOp.Cos => OrtKI.Cos(input),
            UnaryOp.Cosh => OrtKI.Cosh(input),
            UnaryOp.Exp => OrtKI.Exp(input),
            UnaryOp.Floor => OrtKI.Floor(input.Cast(OrtDataType.Float)).Cast(input.DataType),
            UnaryOp.Log => OrtKI.Log(input),
            UnaryOp.Neg => OrtKI.Neg(input),
            UnaryOp.Round => OrtKI.Round(input),
            UnaryOp.Rsqrt => OrtKI.Rsqrt(input),
            UnaryOp.Sin => OrtKI.Sin(input),
            UnaryOp.Sinh => OrtKI.Sinh(input),
            UnaryOp.Sign => OrtKI.Sign(input),
            UnaryOp.Sqrt => OrtKI.Sqrt(input),
            UnaryOp.Square => OrtKI.Square(input),
            UnaryOp.Tanh => OrtKI.Tanh(input),
            UnaryOp.BitwiseNot => throw new NotSupportedException("NotSupported UnaryOp BitwiseNot"),
            UnaryOp.LogicalNot => OrtKI.Not(input),
            _ => throw new ArgumentOutOfRangeException(nameof(unary)),
        };
        return result.ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Unary target)
    {
        var inputType = context.GetArgumentType(target, Unary.Input);

        return inputType switch
        {
            TensorType tensorType => Visit(tensorType),
            DistributedType distTensorType => Visit(distTensorType, target.UnaryOp),
            _ => new InvalidType($"Not support {inputType.GetType().Name}"),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Unary target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Unary.Input);
        var outputType = context.GetReturnType<IRType>();
        return (inputType, outputType) switch
        {
            (TensorType tensorType, TensorType tensorType1) => Visit(tensorType, tensorType1, target),
            (DistributedType distTensorType, DistributedType distTensorType1) => Visit(distTensorType, distTensorType1, target),
            _ => throw new NotSupportedException(string.Empty),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Unary target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Unary.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Unary target, bool iLmode)
    {
        var op_str = target.UnaryOp switch
        {
            UnaryOp.BitwiseNot => "!",
            UnaryOp.LogicalNot => "!",
            var op => op.ToString(),
        };
        if (!iLmode)
        {
            return $"{op_str}({string.Join(", ", target.Parameters.Select(p => p.Name + ": " + context.GetArgument(target, p).Serialize()))})";
        }

        throw new NotSupportedException("ILmode = true");
    }

    public Expr Visit(IShapeEvaluateContext context, Unary target)
    {
        return context.GetArgumentShape(target, Unary.Input);
    }

    private IRType Visit(DistributedType inType, UnaryOp unaryOp)
    {
        var invalid = new InvalidType(inType.ToString());
        var ndsbp = new SBP[inType.Placement.Rank];
        for (int i = 0; i < inType.Placement.Rank; i++)
        {
            if (inType.NdSbp[i] is SBPPartialSum && unaryOp != UnaryOp.Neg)
            {
                return invalid;
            }

            ndsbp[i] = inType.NdSbp[i];
        }

        return new DistributedType(inType.TensorType, ndsbp, inType.Placement);
    }

    private int Compute_int(int input, UnaryOp op) => op switch
    {
        UnaryOp.Ceil => input,
        UnaryOp.Floor => input,
        UnaryOp.Neg => -input,
        UnaryOp.Abs => System.Math.Abs(input),
        UnaryOp.Square => input * input,
        _ => throw new ArgumentOutOfRangeException(nameof(op), $"NotSupported {nameof(op)} For Int"),
    };

    private float Compute_float(float input, UnaryOp op) => op switch
    {
        UnaryOp.Abs => System.MathF.Abs(input),
        UnaryOp.Acos => System.MathF.Acos(input),
        UnaryOp.Acosh => System.MathF.Acosh(input),
        UnaryOp.Asin => System.MathF.Asin(input),
        UnaryOp.Asinh => System.MathF.Asinh(input),
        UnaryOp.Ceil => System.MathF.Ceiling(input),
        UnaryOp.Cos => System.MathF.Cos(input),
        UnaryOp.Cosh => System.MathF.Cosh(input),
        UnaryOp.Exp => System.MathF.Exp(input),
        UnaryOp.Floor => System.MathF.Floor(input),
        UnaryOp.Log => System.MathF.Log(input),
        UnaryOp.Neg => -input,
        UnaryOp.Round => System.MathF.Round(input),
        UnaryOp.Rsqrt => 1.0f / System.MathF.Sqrt(input),
        UnaryOp.Sin => System.MathF.Sin(input),
        UnaryOp.Sinh => System.MathF.Sinh(input),
        UnaryOp.Sign => System.MathF.Sign(input),
        UnaryOp.Sqrt => System.MathF.Sqrt(input),
        UnaryOp.Square => input * input,
        UnaryOp.Tanh => System.MathF.Tanh(input),
        _ => throw new ArgumentOutOfRangeException(nameof(op), $"NotSupported {nameof(op)} For Float"),
    };

    private IRType Visit(TensorType input)
    {
        return input;
    }

    private Cost Visit(DistributedType inType, DistributedType outType, Unary target)
    {
        var inPartType = DistributedUtility.GetDividedTensorType(inType);
        var outPartType = DistributedUtility.GetDividedTensorType(outType);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inPartType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outPartType, CostUtility.GetCPUCyclesOfUnary(target.UnaryOp)),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outPartType),
        };
    }

    private Cost Visit(TensorType inputType, TensorType outputType, Unary target)
    {
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, CostUtility.GetCPUCyclesOfUnary(target.UnaryOp)),
        };
    }
}
