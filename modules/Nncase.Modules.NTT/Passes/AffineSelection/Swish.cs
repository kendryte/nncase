﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectSwish(IR.NN.Swish swish, Call call, Expr output)
    {
        var beta = (Expr)call[IR.NN.Swish.Beta];
        if (output.CheckedShape is not { Rank: > 0 }
            || beta is not TensorConst betaConst)
        {
            return call;
        }

        return SelectUnaryLike((Expr)call[IR.NN.Swish.Input], new TIR.NTT.Swish(betaConst.Value.ToScalar<float>()), call, output);
    }
}
