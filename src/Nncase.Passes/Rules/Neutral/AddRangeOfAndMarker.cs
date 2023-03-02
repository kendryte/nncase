// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Add range of marker base class.
/// </summary>
[RuleGenerator]
public partial class AddRangeOfAndMarker : RewriteRule<Pattern>
{
    private static readonly Dictionary<RuntimeTypeHandle, int> _Dict = new()
    {
        { typeof(GetItem).TypeHandle, 0 },
        { typeof(Transpose).TypeHandle, 1 },
        { typeof(SpaceToBatch).TypeHandle, 1 },
        { typeof(Sigmoid).TypeHandle, 1 },
        { typeof(Relu).TypeHandle, 1 },
        { typeof(Relu6).TypeHandle, 1 },
        { typeof(PRelu).TypeHandle, 1 },
        { typeof(LeakyRelu).TypeHandle, 1 },
        { typeof(Celu).TypeHandle, 1 },
        { typeof(Selu).TypeHandle, 1 },
        { typeof(Elu).TypeHandle, 1 },
        { typeof(HardSwish).TypeHandle, 1 },
        { typeof(HardSigmoid).TypeHandle, 1 },
        { typeof(ResizeImage).TypeHandle, 1 },
        { typeof(ReduceWindow2D).TypeHandle, 1 },
        { typeof(Reduce).TypeHandle, 1 },
        { typeof(Pad).TypeHandle, 1 },
        { typeof(BatchToSpace).TypeHandle, 1 },
        { typeof(Broadcast).TypeHandle, 1 },
        { typeof(LSTM).TypeHandle, 1 },
        { typeof(MatMul).TypeHandle, 2 },
        { typeof(Conv2D).TypeHandle, 2 },
        { typeof(Conv2DTranspose).TypeHandle, 2 },
        { typeof(Compare).TypeHandle, 2 },
        { typeof(Binary).TypeHandle, 2 },
        { typeof(Clamp).TypeHandle, 3 },
    };

    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
      IsCallWildcard(
          "call",
          IsOp<Op>("op"),
          IsWildcard("input"));

    private Expr? GetReplace(Call call, Op op, IReadOnlyList<Expr> callParams, RunPassContext context)
    {
        if (!_Dict.TryGetValue(op.GetType().TypeHandle, out var length))
        {
            return null;
        }

        if (op is Binary binary && (binary.BinaryOp == BinaryOp.LogicalAnd || binary.BinaryOp == BinaryOp.LogicalOr || binary.BinaryOp == BinaryOp.LogicalXor))
        {
            return null;
        }

        if (op is Unary u && u.UnaryOp == UnaryOp.LogicalNot)
        {
            return null;
        }

        var pairs = new List<(Expr, Expr)>();
        for (int i = 0; i < length; i++)
        {
            if (callParams[i] is not Marker)
            {
                pairs.Add((callParams[i], IR.F.Math.RangeOfMarker(callParams[i], IR.F.Math.RangeOf(callParams[i]))));
            }
        }

        if (pairs.Count == 0)
        {
            return null;
        }

        var newCall = ReplaceCallParams(op, callParams, pairs.ToArray());
        return op switch
        {
            LSTM => newCall, // note lstm output can't add marker.
            _ => IR.F.Math.RangeOfMarker(newCall, IR.F.Math.RangeOf(newCall)),
        };
    }
}
