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
using Tensorflow;
using Tensorflow.NumPy;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Transpose"/>.
/// </summary>
public class TransposeEvaluator : IEvaluator<Transpose>, ITypeInferencer<Transpose>, ICostEvaluator<Transpose>,
    IShapeEvaluator<Transpose>
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
        var input = context.CheckArgumentType<TensorType>(target, Transpose.Input);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Transpose target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Transpose.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Transpose target)
    {
        var perm = context.GetArgument(target, Transpose.Perm);
        var rank = context.GetArgument(target, Transpose.Input).CheckedShape.Rank;
        var permValue = perm;
        var inShape = context.GetArgumentShape(target, Transpose.Input);
        var outShape = Enumerable.Range(0, rank).Select(i => inShape[i]).ToArray();
        for (int i = 0; i < rank; i++)
        {
            outShape[i] = inShape[permValue[i]];
        }

        return IR.F.Tensors.Stack(new IR.Tuple(outShape), 0);
    }

    private IRType Visit(ITypeInferenceContext context, Transpose target, TensorType input)
    {
        var permExpr = context.GetArgument(target, Transpose.Perm);
        return TypeInference.TransposeType(input, permExpr);
    }
}
