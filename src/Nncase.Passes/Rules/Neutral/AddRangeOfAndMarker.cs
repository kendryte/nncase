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
    private static readonly Dictionary<RuntimeTypeHandle, int> _DictRange = new()
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
        { typeof(Swish).TypeHandle, 1 },
        { typeof(HardSigmoid).TypeHandle, 1 },
        { typeof(Erf).TypeHandle, 1 },
        { typeof(Gelu).TypeHandle, 1 },
        { typeof(ResizeImage).TypeHandle, 1 },
        { typeof(ReduceWindow2D).TypeHandle, 1 },
        { typeof(Reduce).TypeHandle, 1 },
        { typeof(Pad).TypeHandle, 1 },
        { typeof(BatchToSpace).TypeHandle, 1 },
        { typeof(Broadcast).TypeHandle, 1 },
        { typeof(Unary).TypeHandle, 1 },
        { typeof(MatMul).TypeHandle, 2 },
        { typeof(Conv2D).TypeHandle, 2 },
        { typeof(Conv2DTranspose).TypeHandle, 2 },
        { typeof(Compare).TypeHandle, 2 },
        { typeof(Binary).TypeHandle, 2 },
        { typeof(Clamp).TypeHandle, 3 },
    };

    private static readonly Dictionary<RuntimeTypeHandle, int[]> _DictList = new() { { typeof(LSTM).TypeHandle, new[] { 0, 1, 2, 5, 6 } }, };

    private static readonly Pattern _pattern = IsCallWildcard(
                "call",
                IsOp<Op>("op"),
                IsWildcard("input")) with
    {
        TypePattern = HasDataType(DataTypes.Float32) | IsTuple(t => t.All(tt => tt is TensorType { DType: DataType dt } && dt == DataTypes.Float32), "AllElementsAreF32"),
    };

    /// <inheritdoc/>
    public override Pattern Pattern => _pattern;

    /// <summary>
    /// check op.
    /// </summary>
    /// <param name="op">op.</param>
    /// <returns>can add the marker.</returns>
    public static bool CheckOp(Op op)
    {
        if (op is Binary binary && (binary.BinaryOp == BinaryOp.LogicalAnd || binary.BinaryOp == BinaryOp.LogicalOr || binary.BinaryOp == BinaryOp.LogicalXor))
        {
            return false;
        }

        if (op is Unary u && u.UnaryOp == UnaryOp.LogicalNot)
        {
            return false;
        }

        return true;
    }

    private Expr? GetReplace(Call call, Op op, IReadOnlyList<Expr> callParams, RunPassContext context)
    {
        int length = 0;
        _ = Array.Empty<int>();
        int[]? list;
        if (!_DictList.TryGetValue(op.GetType().TypeHandle, out list) && !_DictRange.TryGetValue(op.GetType().TypeHandle, out length))
        {
            return null;
        }

        if (!CheckOp(op))
        {
            return null;
        }

        var pairs = new Dictionary<Expr, Expr>();
        if (list is null)
        {
            list = Enumerable.Range(0, length).ToArray();
        }

        foreach (var i in list)
        {
            if (callParams[i] is not Marker)
            {
                if (!pairs.ContainsKey(callParams[i]))
                {
                    pairs.Add(callParams[i], IR.F.Math.RangeOfMarker(callParams[i], IR.F.Math.RangeOf(callParams[i])));
                }
            }
        }

        Call newCall;
        if (pairs.Count != 0)
        {
            newCall = ReplaceCallParams(op, callParams, list.Where(i => callParams[i] is not Marker).Select(i => (call: callParams[i], pairs[callParams[i]])).ToArray());
        }
        else
        {
            newCall = new Call(op, callParams.ToArray());
        }

        context.MatchOptions.SuppressPattern(newCall, Pattern);
        return op switch
        {
            LSTM => WrapLSTMOutput(newCall, ((TensorConst)newCall[LSTM.OutputSize]).Value.ToScalar<int>()), // note lstm output can't add marker.
            _ => IR.F.Math.RangeOfMarker(newCall, IR.F.Math.RangeOf(newCall)),
        };
    }

    private IR.Tuple WrapLSTMOutput(Call call, int outputSize)
    {
        var outputs = Enumerable.Range(0, outputSize).Select(i => IR.F.Tensors.GetItem(call, i)).ToArray();
        var exprs = outputs.Select(item => IR.F.Math.RangeOfMarker(item, IR.F.Math.RangeOf(item))).ToArray();
        return new IR.Tuple(exprs);
    }
}
