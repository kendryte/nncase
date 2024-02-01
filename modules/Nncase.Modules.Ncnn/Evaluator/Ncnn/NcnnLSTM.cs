// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.RNN;
using Nncase.IR.Ncnn;
using OrtKISharp;
using static Nncase.LSTMHelper;
namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnLSTM"/>.
/// </summary>
public class NcnnLSTMEvaluator : IEvaluator<NcnnLSTM>, ITypeInferencer<NcnnLSTM>, ICostEvaluator<NcnnLSTM>, IShapeEvaluator<NcnnLSTM>, IMetricEvaluator<NcnnLSTM>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnLSTM lstm)
    {
        var input = context.GetOrtArgumentValue(lstm, NcnnLSTM.Input);
        // TODO: reorder w,r to iofg, b to numDirection-8*hiddensize (Now, numDirection-4*hiddensize (bw+br))
        var w = lstm.W;
        var r = lstm.R;
        var b = lstm.B;
        var direction_ = lstm.Direction > 1 ? 2 : 1;
        var mode = direction_ switch
        {
            0 => LSTMDirection.Forward,
            1 => LSTMDirection.Reverse,
            2 => LSTMDirection.Bidirectional,
            _ => throw new NotImplementedException(),
        };

        var seqLens = (int)input.Shape[0];
        var initH = Tensor.Zeros<float>(new[] { direction_, seqLens, lstm.HiddenSize});
        var initC = Tensor.Zeros<float>(new[] { direction_, seqLens, lstm.HiddenSize });
        var p = Tensor.Zeros<float>(new[] { 1, seqLens, lstm.HiddenSize });
        var actAlpha = new[] { 0.0f };
        var actBeta = new[] { 0.0f };
        var clip = float.NaN;
        var hiddenSize = lstm.HiddenSize;
        var inputForget = 1;
        var outputSize = 1;
        var result = OrtKI.LSTM(input, w, r, b, seqLens, initH.ToOrtTensor(), initC.ToOrtTensor(), p.ToOrtTensor(), actAlpha, actBeta, new string[] { "sigmoid", "sigmoid", "tanh" }, clip, LSTMDirectionToValue(mode), hiddenSize, inputForget, 0, !clip.Equals(float.NaN), outputSize);
        return Value.FromTensors(result.Select(t => t.ToTensor()).ToArray());

    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnLSTM target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnLSTM.Input);
        // [num_direction, batch_size:1, hidden_size]
        var initH = new TensorType(DataTypes.Float32, new[] { target.Direction > 1 ? 2 : 1, 1, target.HiddenSize });
        var initC = new TensorType(DataTypes.Float32, new[] { target.Direction > 1 ? 2 : 1, 1, target.HiddenSize });
        if (input.Shape.Rank != 3)
        {
            // TODO: confirm ncnn input dims when direction not 1.
            input.Shape.InsertAndClone(1, 1);
        }

        return Visit(context, input, initH, initC, target.OutputSize);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnLSTM target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnLSTM target_)
    {
        var direction_ = target_.Direction > 1 ? 2 : 1;

        var xType = context.GetArgumentType<TensorType>(target_, NcnnLSTM.Input);
        var wType = new TensorType(DataTypes.Float32, new[] { direction_, 4 * target_.HiddenSize, xType.Shape[^1] });
        var rType = new TensorType(DataTypes.Float32, new[] { direction_, 4 * target_.HiddenSize, target_.HiddenSize });
        var bType = new TensorType(DataTypes.Float32, new[] { direction_, 8 * target_.HiddenSize });
        var returnType = context.GetReturnType<TupleType>();
        var outputYType = (TensorType)returnType[0];
        var outputYShape = outputYType.Shape.ToValueArray().Select(s => (UInt128)s).ToArray();
        var (sequence_len, num_directions, batch_size, hidden_size) = (outputYShape[0], outputYShape[1], outputYShape[2], outputYShape[3]);
        var embbeding_size = (UInt128)xType.Shape[^1].FixedValue;

        var flops = num_directions * batch_size * sequence_len * (
            MetricUtility.GetMatMulFLOPs(1, 4 * hidden_size, embbeding_size)

            // [1,embbeding_size] @ [embbeding_size, 4 * hidden_size]
            + MetricUtility.GetMatMulFLOPs(1, 4 * hidden_size, hidden_size) // [1,hidden_size] @ [hidden_size, 4 * hidden_size]
            + (4 * hidden_size)
            + (MetricUtility.SigmoidFLOPs * hidden_size) // ft = sigmoid(g[2])
            + hidden_size // ct = init_c * ft
            + (MetricUtility.SigmoidFLOPs * hidden_size) // it = sigmoid(g[0])
            + (MetricUtility.TanhFLOPs * hidden_size) // c_t = tanh(g[3])
            + hidden_size // c_t_it = it * c_t
            + hidden_size // ct = ct + c_t_it
            + (MetricUtility.SigmoidFLOPs * hidden_size) // ot = sigmoid(g[1])
            + (MetricUtility.TanhFLOPs * hidden_size) // tanh_ct = tanh(ct_o)
            + hidden_size); // ht = tanh_ct * ot

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(xType) + CostUtility.GetMemoryAccess(wType) + CostUtility.GetMemoryAccess(rType) + CostUtility.GetMemoryAccess(bType) + returnType.Select(t => t switch
            {
                TensorType tensorType => CostUtility.GetMemoryAccess(tensorType),
                _ => UInt128.One,
            }).Sum(),
            [MetricFactorNames.FLOPs] = flops,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnLSTM target) => context.GetArgumentShape(target, NcnnLSTM.Input);

    private IRType Visit(ITypeInferenceContext context, TensorType x, TensorType initH, TensorType initC, int outputSize)
    {
        // TODO: confirm ncnn output
        var numDirections = initH.Shape[0];
        var seqLenIndex = 0;
        var yType = new TensorType(DataTypes.Float32, new[] { x.Shape[0].FixedValue, initH.Shape[2].FixedValue });
        Console.WriteLine($"x.Shape[0].FixedValue: {x.Shape[0].FixedValue},  initH.Shape[2].FixedValue:{initH.Shape[2].FixedValue}");
        var result = new[] { yType, initH, initC };
        return new TupleType(result[..outputSize]);
    }
}
