﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="LogSoftmax"/>.
/// </summary>
public class LogSoftmaxEvaluator : IEvaluator<LogSoftmax>, ITypeInferencer<LogSoftmax>, ICostEvaluator<LogSoftmax>, IShapeEvaluator<LogSoftmax>, IMetricEvaluator<LogSoftmax>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, LogSoftmax logSoftMax)
    {
        var input = context.GetOrtArgumentValue(logSoftMax, LogSoftmax.Input);
        var axis = context.GetArgumentValueAsScalar<long>(logSoftMax, LogSoftmax.Axis);
        return OrtKI.LogSoftmax(input, axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, LogSoftmax target)
    {
        var input = context.CheckArgumentType<TensorType>(target, LogSoftmax.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, LogSoftmax target)
    {
        var ret = context.GetReturnType<TensorType>();
        uint macPerElement = 4;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, macPerElement),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, LogSoftmax target) => context.GetArgumentShape(target, LogSoftmax.Input);

    public Metric Visit(IMetricEvaluateContext context, LogSoftmax target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, LogSoftmax.Input);
        var returnType = context.GetReturnType<TensorType>();
        var returnF = MetricUtility.GetFLOPs(returnType);
        var inputF = MetricUtility.GetFLOPs(inputType);
        var inner = inputF / returnF;

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = (inner * 2) + (inputF * (MetricUtility.SubFLOPs + MetricUtility.ExpFLOPs + MetricUtility.DivFLOPs + MetricUtility.LogFLOPs)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Softmax"/>.
/// </summary>
public class SoftmaxEvaluator : IEvaluator<Softmax>, ITypeInferencer<Softmax>, ICostEvaluator<Softmax>, IShapeEvaluator<Softmax>, IMetricEvaluator<Softmax>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Softmax softMax)
    {
        var input = context.GetOrtArgumentValue(softMax, Softmax.Input);
        var dim = context.GetArgumentValueAsScalar<int>(softMax, Softmax.Axis);
        return OrtKI.Softmax(input, dim).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Softmax target)
    {
        var input = context.CheckArgumentType<IRType>(target, Softmax.Input);
        var axis = context.GetArgument(target, Softmax.Axis);
        return input switch
        {
            TensorType t => Visit(t),
            DistributedType d => Visit(d, axis),
            _ => new InvalidType(input.GetType().Name),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Softmax target)
    {
        var ret = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Softmax target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Softmax.Input);
        var returnType = context.GetReturnType<TensorType>();
        var returnF = MetricUtility.GetFLOPs(returnType);
        var inputF = MetricUtility.GetFLOPs(inputType);
        var inner = inputF / returnF;

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = (inner * 2) + (inputF * (MetricUtility.SubFLOPs + MetricUtility.ExpFLOPs + MetricUtility.DivFLOPs)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Softmax target) => context.GetArgumentShape(target, Softmax.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }

    private IRType Visit(DistributedType input, Expr axisExpr)
    {
        var axis = ((TensorConst)axisExpr).Value.ToScalar<int>();
        if (input.NdSBP.Any(sbp => sbp is SBPSplit s && s.Axis == axis))
        {
            return new InvalidType("Not support split on Axis for Softmax now.");
        }

        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Softplus"/>.
/// </summary>
public class SoftplusEvaluator : IEvaluator<Softplus>, ITypeInferencer<Softplus>, ICostEvaluator<Softplus>, IMetricEvaluator<Softplus>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Softplus softPlus)
    {
        var input = context.GetOrtArgumentValue(softPlus, Softplus.Input);
        return OrtKI.Softplus(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Softplus target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Softplus.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Softplus target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Softplus target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Softplus.Input);
        var returnType = context.GetReturnType<TensorType>();
        var r = MetricUtility.GetFLOPs(returnType);
        var i = MetricUtility.GetFLOPs(inputType);

        var reduced = r / i;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = (i * ((MetricUtility.ExpFLOPs * 2) + 3)) + reduced,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Softsign"/>.
/// </summary>
public class SoftsignEvaluator : IEvaluator<Softsign>, ITypeInferencer<Softsign>, ICostEvaluator<Softsign>, IMetricEvaluator<Softsign>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Softsign softSign)
    {
        var input = context.GetOrtArgumentValue(softSign, Softsign.Input);
        return OrtKI.Softsign(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Softsign target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Softsign.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Softsign target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Softsign target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Softsign.Input);
        var returnType = context.GetReturnType<TensorType>();
        var r = MetricUtility.GetFLOPs(returnType);
        var i = MetricUtility.GetFLOPs(inputType);

        var reduced = r / i;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = (i * ((MetricUtility.ExpFLOPs * 2) + 3)) + reduced,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
