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
    IMetricEvaluator<Transpose>
{
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
            var ndsbp = new SBP[tensorType.Shape.Rank];

            for (int i = 0; i < ndsbp.Length; i++)
            {
                ndsbp[i] = input.AxisPolices[perm[i]];
            }

            return new DistributedType(tensorType, ndsbp, input.Placement);
        }

        return new InvalidType(input.ToString());
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Transpose tr)
    {
        var inputValue = context.GetArgumentValue(tr, Transpose.Input);
        var perm = context.GetArgumentValueAsArray<long>(tr, Transpose.Perm);
        OrtKISharp.Tensor input;
        var inputOrg = inputValue.AsTensor();
        var dataType = inputOrg.ElementType;

        if (dataType is VectorType { ElemType: DataType dataTypes } vType && dataTypes != DataTypes.Float32)
        {
            var interType = new VectorType(DataTypes.Float32, vType.Lanes);
            input = Nncase.IR.F.Tensors.Cast(inputOrg, interType).Evaluate().AsTensor().ToOrtTensor();
        }
        else if (dataType is not VectorType && dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            input = Nncase.IR.F.Tensors.Cast(inputOrg, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
        }
        else
        {
            input = context.GetOrtArgumentValue(tr, Transpose.Input);
        }

        if (inputValue.Type is TensorType { DType: VectorType vectorType })
        {
            var newPerm = perm.Concat(Enumerable.Range(0, vectorType.Lanes.Count).Select(i => (long)(i + perm.Length))).ToArray();
            var output = OrtKI.Transpose(input, newPerm);
            return Value.FromTensor(Tensor.FromBytes(new TensorType(vectorType, output.Shape.Select(i => (int)i).SkipLast(vectorType.Lanes.Count).ToArray()), output.BytesBuffer.ToArray()));
        }

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

        if (dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            return Value.FromTensor(OrtKI.Transpose(input, perm).ToTensor().CastTo(dataType));
        }
        else
        {
            return OrtKI.Transpose(input, perm).ToValue();
        }
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
}
