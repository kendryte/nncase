// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR;
using OrtKISharp;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator;

public static class EvaluatorUtil
{
    /// <summary>
    /// nncase pads format to onnx pads format.
    /// </summary>
    public static long[] ToOnnxPadFormat(OrtKISharp.Tensor pads)
    {
        if (pads.Rank != 2)
        {
            throw new InvalidOperationException($"Pad's rank must be 2, but get {pads.Rank}");
        }

        // note the pads will be int or long, need cast to long
        return OrtKI.Transpose(pads.Cast(OrtDataType.Int64), new long[] { 1, 0 }).ToArray<long>();
    }

    public static Dictionary<Expr, IValue> GetMemo(Expr input, Dictionary<Var, IValue> varValues)
    {
        var visitor = new EvaluateVisitor(varValues, new());
        visitor.Visit(input);
        return visitor.ExprMemo;
    }
}
