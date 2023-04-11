// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Utilities.DumpUtility;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestEvaluatorRNN : TestClassBase
{
    [Fact]
    public void TestLSTM()
    {
        var inputSize = 2;
        var hiddenSize = 1;
        var outputSize = 1;
        var direction = LSTMDirection.Forward;
        var batchSize = 1;
        var seqLength = 1;
        var numberDirections = 1;
        var x = OrtKI.Random(seqLength, batchSize, inputSize);
        var initC = OrtKI.Random(numberDirections, batchSize, hiddenSize);
        var initH = OrtKI.Random(numberDirections, batchSize, hiddenSize);
        var b = OrtKI.Random(numberDirections, 8 * hiddenSize);
        var w = OrtKI.Random(numberDirections, 4 * hiddenSize, inputSize);
        var r = OrtKI.Random(numberDirections, 4 * hiddenSize, hiddenSize);
        var p = new float[numberDirections, 3 * hiddenSize];
        var acts = new[] { "Sigmoid", "Tanh", "Tanh" };
        var expr = IR.F.RNN.LSTM(direction, LSTMLayout.Zero, acts, x.ToTensor(), w.ToTensor(), r.ToTensor(), b.ToTensor(), new[] { seqLength }, initH.ToTensor(), initC.ToTensor(), p, 0, 0, float.NaN, hiddenSize, 0, outputSize);
        CompilerServices.InferenceType(expr);
        var expect = OrtKI.LSTM(x, w, r, b, Tensor.FromArray(new[] { seqLength }).ToOrtTensor(), initH, initC, Tensor.FromArray(p).ToOrtTensor(), new[] { 0f }, new[] { 0f }, acts, float.NaN, LSTMHelper.LSTMDirectionToValue(direction), hiddenSize, 0, LSTMHelper.LSTMLayoutToValue(LSTMLayout.Zero), false, outputSize);
        Assert.Equal(expect[0], expr.Evaluate().AsTensors()[0].ToOrtTensor());
    }
}
