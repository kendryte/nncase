// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.IR.RNN;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.RNN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerLSTM : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsLSTM(
        target_name: "lstm",
        call_name: "call",
        _ => true,
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("w"),
        IsTensorConst("r"),
        IsTensorConst("b"),
        IsTensorConst(),
        IsTensorConst("initialH"),
        IsTensorConst("initialC"),
        IsTensorConst(),
        IsTensorConst(),
        IsTensorConst(),
        IsTensorConst(),
        IsTensorConst("hiddenSize"),
        IsTensorConst(),
        IsTensorConst("outputSize"));

    private Expr? GetReplace(LSTM lstm, Call call, Expr input, Tensor<float> w, Tensor<float> r, Tensor<float> b, Expr initialH, Expr initialC, int hiddenSize, int outputSize)
    {
        // x 先Squeeze掉batch size   axis=1，
        // ncnn : initial(H,C)默认为0，不起作用
        var inRes = Squeeze(input, new[] { 1 });
        var inResO = new Var(inRes.CheckedType);

        // args
        // 0: hiddensize
        // 1: weight_data_size : w.count/num_direction
        var weightDataSize = w.Shape[1].FixedValue * w.Shape[2].FixedValue;

        // 2: direction [0,1,2]
        int direction;
        switch (lstm.Direction)
        {
            case LSTMDirection.Forward:
                direction = 0;
                break;
            case LSTMDirection.Reverse:
                direction = 1;
                break;
            default: // Bidirectional
                direction = 2;
                break;
        }

        // bin
        // reorder num_directions-IOFG - hidden_size to num_directions-IFOG - hidden_size
        // weights: i f o g:  + reverse i f o g
        var (n_, c_, h_, w_) = (w.Shape[0].FixedValue, 4, hiddenSize, w.Shape[2].FixedValue);
        var newWeights_ = Reshape(w, new[] { n_, c_, h_, w_ });
        var strides = new int[1] { 1 };
        var newWeights = Concat(
                        new IR.Tuple(new[]
                        {
                            Slice(newWeights_, new int[] { 0 }, new int[] { 1 }, new int[] { 1 }, strides),
                            Slice(newWeights_, new int[] { 2 }, new int[] { 3 }, new int[] { 1 }, strides),
                            Slice(newWeights_, new int[] { 1 }, new int[] { 2 }, new int[] { 1 }, strides),
                            Slice(newWeights_, new int[] { 3 }, new int[] { 4 }, new int[] { 1 }, strides),
                        }),
                        1).Evaluate().AsTensor().ToArray<float>();

        // bias : x_b + w_b_re
        var newBias_ = Reshape(b, new[] { w.Shape[0].FixedValue, 2, 4, hiddenSize });
        var biasW_ = Slice(newBias_, new[] { 0 }, new[] { 1 }, new int[] { 1 }, strides);
        var biasR_ = Slice(newBias_, new[] { 1 }, new[] { 2 }, new int[] { 1 }, strides);
        var biasW = Concat(
                        new IR.Tuple(new[]
                        {
                            Slice(biasW_, new int[] { 0 }, new int[] { 1 }, new int[] { 2 }, strides),
                            Slice(biasW_, new int[] { 2 }, new int[] { 3 }, new int[] { 2 }, strides),
                            Slice(biasW_, new int[] { 1 }, new int[] { 2 }, new int[] { 2 }, strides),
                            Slice(biasW_, new int[] { 3 }, new int[] { 4 }, new int[] { 2 }, strides),
                        }),
                        2);
        var biasR = Concat(
                        new IR.Tuple(new[]
                        {
                            Slice(biasR_, new int[] { 0 }, new int[] { 1 }, new int[] { 2 }, strides),
                            Slice(biasR_, new int[] { 2 }, new int[] { 3 }, new int[] { 2 }, strides),
                            Slice(biasR_, new int[] { 1 }, new int[] { 2 }, new int[] { 2 }, strides),
                            Slice(biasR_, new int[] { 3 }, new int[] { 4 }, new int[] { 2 }, strides),
                        }),
                        2);
        float[] newBias = (biasW + biasR).Evaluate().AsTensor().ToArray<float>();

        // R: 处理同weights
        var newR_ = Reshape(r, new[] { n_, c_, h_, hiddenSize });
        float[] newR = Concat(
                        new IR.Tuple(new[]
                        {
                            Slice(newR_, new int[] { 0 }, new int[] { 1 }, new int[] { 1 }, strides),
                            Slice(newR_, new int[] { 2 }, new int[] { 3 }, new int[] { 1 }, strides),
                            Slice(newR_, new int[] { 1 }, new int[] { 2 }, new int[] { 1 }, strides),
                            Slice(newR_, new int[] { 3 }, new int[] { 4 }, new int[] { 1 }, strides),
                        }),
                        1).Evaluate().AsTensor().ToArray<float>();

        var lstm_ = new Call(new Fusion("ncnn", NcnnLSTM(inResO, outputSize, hiddenSize, weightDataSize, direction, newWeights, newBias, newR), new[] { inResO }), inRes);
        return new IR.Tuple(new[] { Unsqueeze(lstm_[0], new[] { 1, 1 }), lstm_[1], lstm_[2] });
    }
}
