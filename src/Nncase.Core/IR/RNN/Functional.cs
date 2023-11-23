// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.NN;
using Nncase.IR.Random;
using Nncase.IR.Tensors;

namespace Nncase.IR.F;

/// <summary>
/// RNN functional helper.
/// </summary>
public static class RNN
{
    public static Call LSTM(LSTMDirection direction, LSTMLayout layout, string[] acts, Expr x, Expr w, Expr r, Expr b, Expr seqLens, Expr initH, Expr initC, Expr p, Expr actAlpha, Expr actBeta, Expr clip, Expr hiddenSize, Expr inputForget, Expr outputSize) =>
        new Call(new IR.RNN.LSTM(direction, layout, acts), x, w, r, b, seqLens, initH, initC, p, actAlpha, actBeta, clip, hiddenSize, inputForget, outputSize);
}
