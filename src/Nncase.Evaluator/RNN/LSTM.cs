// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;

// using Nncase.IR.NN;
using Nncase.IR.Tensors;
using OrtKISharp;
using static Nncase.LSTMHelper;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="LSTM"/>.
/// </summary>
public class LSTMEvaluator : IEvaluator<LSTM>, ITypeInferencer<LSTM>, ICostEvaluator<LSTM>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, LSTM target)
    {
        var x = context.GetOrtArgumentValue(target, LSTM.X);
        var w = context.GetOrtArgumentValue(target, LSTM.W);
        var r = context.GetOrtArgumentValue(target, LSTM.R);
        var b = context.GetOrtArgumentValue(target, LSTM.B);
        var seqLens = context.GetOrtArgumentValue(target, LSTM.SequenceLens);
        var initH = context.GetOrtArgumentValue(target, LSTM.InitialH);
        var initC = context.GetOrtArgumentValue(target, LSTM.InitialC);
        var p = context.GetOrtArgumentValue(target, LSTM.P);
        var actAlpha = context.GetArgumentValueAsArray<float>(target, LSTM.ActivationAlpha);
        var actBeta = context.GetArgumentValueAsArray<float>(target, LSTM.ActivationBeta);
        var clip = context.GetArgumentValueAsScalar<float>(target, LSTM.Clip);
        var hiddenSize = context.GetArgumentValueAsScalar<long>(target, LSTM.HiddenSize);
        var inputForget = context.GetArgumentValueAsScalar<long>(target, LSTM.InputForget);
        var outputSize = context.GetArgumentValueAsScalar<long>(target, LSTM.OutputSize);
        var result = OrtKI.LSTM(x, w, r, b, seqLens, initH, initC, p, actAlpha, actBeta, target.Activations.ToArray(), clip, LSTMDirectionToValue(target.Direction), hiddenSize, inputForget, LSTMLayoutToValue(target.Layout), !clip.Equals(float.NaN), outputSize);
        return Value.FromTensors(result.Select(t => t.ToTensor()).ToArray());
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, LSTM target)
    {
        var x = context.CheckArgumentType<TensorType>(target, LSTM.X);
        var initH = context.CheckArgumentType<TensorType>(target, LSTM.InitialH);
        var initC = context.CheckArgumentType<TensorType>(target, LSTM.InitialC);
        if (x.Shape.Rank != 3)
        {
            return new InvalidType("LSTM First input tensor must have rank 3");
        }

        return Visit(context, x, initH, initC, target);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, LSTM target)
    {
        var xType = context.GetArgumentType<TensorType>(target, LSTM.X);
        var wType = context.GetArgumentType<TensorType>(target, LSTM.W);
        var rType = context.GetArgumentType<TensorType>(target, LSTM.R);
        var bType = context.GetArgumentType<TensorType>(target, LSTM.B);
        var returnType = context.GetReturnType<TupleType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(xType) + CostUtility.GetMemoryAccess(wType) + CostUtility.GetMemoryAccess(rType) + CostUtility.GetMemoryAccess(bType),
            [CostFactorNames.MemoryStore] = returnType.Select(t => t switch
            {
                TensorType tensorType => CostUtility.GetMemoryAccess(tensorType),
                _ => 1,
            }).Sum(),
        };
    }

    private IRType Visit(ITypeInferenceContext context, TensorType x, TensorType initH, TensorType initC, LSTM target)
    {
        var numDirections = target.Direction == LSTMDirection.Bidirectional ? 2 : 1;
        var seqLenIndex = target.Layout == LSTMLayout.Zero ? 0 : 1;
        if (context.GetArgument(target, LSTM.OutputSize) is TensorConst outSizeConst)
        {
            var yType = InferYType(context, target, x, seqLenIndex, numDirections);
            var result = new[] { yType, initH, initC };
            return new TupleType(result[..outSizeConst.Value.ToScalar<int>()]);
        }
        else
        {
            return new InvalidType("LSTM OutputSize Must be known");
        }
    }

    private TensorType InferYType(ITypeInferenceContext context, LSTM target, TensorType x, int seqLenIndex, int numDirections)
    {
        // layout 0:
        // [seq_length, num_directions, batch_size, hidden_size]
        // layout 1:
        // [batch_size, seq_length, num_directions, hidden_size]
        var yShape = x.Shape.ToList();
        yShape.Insert(seqLenIndex + 1, numDirections);
        var hiddenSize = Dimension.Unknown;
        if (context.GetArgument(target, LSTM.HiddenSize) is TensorConst hiddenSizeConst)
        {
            hiddenSize = hiddenSizeConst.Value.ToScalar<int>();
        }

        yShape[^1] = hiddenSize;
        return x with { Shape = yShape.ToArray() };
    }
}
