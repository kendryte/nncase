// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Transpose"/>.
/// </summary>
public class TransposeEvaluator : IEvaluator<Transpose>, ITypeInferencer<Transpose>, ICostEvaluator<Transpose>,
    IShapeEvaluator<Transpose>, IMetricEvaluator<Transpose>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Transpose tr)
    {
        var input = context.GetOrtArgumentValue(tr, Transpose.Input);
        var perm = context.GetArgumentValueAsArray<long>(tr, Transpose.Perm);

        // when HasBindedMixQuantInfo is true, eval will do simulation of quant/dequant for some inputs, this is used for evaluate accumulated quant error for layers.
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            var pattern = IsRangeOfMarker(IsWildcard(), IsWildcard());
            if (pattern.MatchLeaf(context.CurrentCall.Arguments.ToArray()[0]) &&
                ((Nncase.IR.Marker)context.CurrentCall.Arguments.ToArray()[0]).MixQuantInfo?.HasBindedMixQuantInfo ==
                true)
            {
                var quantParam = ((Nncase.IR.Marker)context.CurrentCall.Arguments.ToArray()[0]).MixQuantInfo!
                    .QuantParameter;

                // input feature map quantParam count should be 1 since input feature map quant is by tensor.
                Trace.Assert(quantParam.Count == 1);
                var inputFloat = input.ToArray<float>();
                for (var i = 0; i < inputFloat.Length; i++)
                {
                    var inputBufQuant =
                        (double)((inputFloat[i] / (double)quantParam[0].Scale) + quantParam[0].ZeroPoint);
                    if (!(quantParam[0].Scale == 1.0f && quantParam[0].ZeroPoint == 0))
                    {
                        inputBufQuant = System.Math.Round((double)(float)inputBufQuant);
                    }

                    var inputBufDeQuant =
                        (float)((inputBufQuant - quantParam[0].ZeroPoint) * (double)quantParam[0].Scale);
                    inputFloat[i] = (float)inputBufDeQuant;
                }

                input = OrtKISharp.Tensor.MakeTensor(inputFloat, input.Shape);
            }
        }

        return OrtKI.Transpose(input, perm).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Transpose target)
    {
        var input = context.CheckArgumentType<IRType>(target, Transpose.Input);
        var permExpr = context.GetArgument(target, Transpose.Perm);

        return input switch
        {
            DistributedType d => Visit(d, permExpr),
            TensorType t => Visit(t, permExpr),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Transpose target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Transpose.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Transpose target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Transpose target)
    {
        var inShape = context.GetArgumentShape(target, Transpose.Input);
        var perm = context.GetArgument(target, Transpose.Perm);
        return IR.F.ShapeExpr.TransposeShape(inShape, perm);
    }

    public static IRType Visit(TensorType input, Expr permExpr)
    {
        return TypeInference.TransposeType(input, permExpr);
    }

    public static IRType Visit(DistributedType input, Expr permExpr)
    {
        if (Visit(input.TensorType, permExpr) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        if (permExpr is TensorConst permValue)
        {
            var perm = permValue.Value.ToArray<int>();
            var ndsbp = new SBP[input.Placement.Rank];

            for (int i = 0; i < input.Placement.Rank; i++)
            {
                switch (input.NdSBP[i])
                {
                    case SBPSplit { Axis: int ix }:
                        ndsbp[i] = SBP.S(perm.IndexOf(ix));
                        break;
                    default:
                        ndsbp[i] = input.NdSBP[i];
                        break;
                }
            }

            return new DistributedType(tensorType, ndsbp, input.Placement);
        }

        return new InvalidType(input.ToString());
    }
}
