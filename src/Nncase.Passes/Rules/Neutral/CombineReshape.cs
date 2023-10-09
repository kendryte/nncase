// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Tuple = System.Tuple;
using Unary = Nncase.IR.Math.Unary;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Combine Reshape with Binary
/// binary(reshape(a,shape),reshape(b,shape)) => reshape(binary(a,b),shape).
/// </summary>
[RuleGenerator]
public sealed partial class CombineBinaryReshape : IRewriteRule
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CombineBinaryReshape"/> class.
    /// </summary>
    public CombineBinaryReshape()
    {
        var shape = IsWildcard("shape");
        Pattern = IsBinary("binary", "call", x => true, IsReshape(IsWildcard("x"), shape), IsReshape(IsWildcard("y"), shape));
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; init; }

    private Expr? GetReplace(Binary binary, Call call, Expr x, Expr y, Expr shape)
    {
        if (x.CheckedShape == y.CheckedShape)
        {
            return Reshape(Binary(binary.BinaryOp, x, y).InheritMetaData(call), shape);
        }

        return null;
    }
}

/// <summary>
/// Combine Reshape with Const Binary, if Const has rank 1.
/// binary(resahpe(a,shape),const(b)) => shape(binary(a,const(b)),shape) or binary(const(a),reshape(b,shape)) => reshape(binary(const(a),b),shape).
/// </summary>
[RuleGenerator]
public sealed partial class CombineConstBinaryReshape : IRewriteRule
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CombineConstBinaryReshape"/> class.
    /// </summary>
    public CombineConstBinaryReshape()
    {
        var shape = IsTensorConst("shape");
        var input = IsReshape(IsWildcard("input"), shape);
        var @const = IsConst("constInput") with { TypePattern = HasRank(1) | HasRank(0) };
        Pattern = IsAlt(IsCallWildcard("call", IsOp<Binary>("binary"), input, @const), IsCallWildcard("call", IsOp<Binary>("binary"), @const, input));
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; init; }

    private Expr? GetReplace(Binary binary, Call call, IReadOnlyList<Expr> callParams, Expr input, TensorConst constInput, TensorConst shape)
    {
        var oldShape = shape.Value.ToArray<int>();
        var significantShape = oldShape.Where(x => x > 1).ToArray();

        bool leftConst = ReferenceEquals(callParams[0], constInput);
        if (constInput.CheckedShape.Rank == 0)
        {
            var res = Reshape(Binary(binary.BinaryOp, leftConst ? constInput : input, leftConst ? input : constInput).InheritMetaData(call), shape);
            res.InferenceType();
            return res;
        }

        if (constInput.CheckedShape.Rank == 1)
        {
            var significantInputShape = input.CheckedShape.ToValueArray().Where(x => x > 1).ToArray();
            var constSize = constInput.CheckedShape.ToValueArray()[0];

            if (significantShape.SequenceEqual(significantInputShape) && oldShape.Length > 0 && oldShape[^1] == constSize)
            {
                var broadcastIndex = Array.LastIndexOf(input.CheckedShape.ToValueArray(), constSize);
                var newConstShape = Enumerable.Repeat(1, input.CheckedShape.Rank - 1 - broadcastIndex).ToList();
                newConstShape.Insert(0, constSize);

                var res = Reshape(Binary(binary.BinaryOp, leftConst ? Reshape(constInput, newConstShape.ToArray()) : input, leftConst ? input : Reshape(constInput, newConstShape.ToArray())).InheritMetaData(call), call.CheckedShape);
                res.InferenceType();
                return res;
            }
        }

        return null;
    }
}

/// <summary>
/// unary(reshape(input, shape) => reshape(unary(input), shape).
/// </summary>
[RuleGenerator]
public sealed partial class CombineUnaryReshape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsUnary(
            "unary",
            "call",
            _ => true,
            IsReshape(IsWildcard("input"), IsWildcard("shape")));

    private Expr? GetReplace(Unary unary, Call call, Expr input, Expr shape)
    {
        return Reshape(
            Unary(unary.UnaryOp, input).InheritMetaData(call),
            shape);
    }
}

/// <summary>
/// activations(reshape(input, shape), args...) => reshape(activations(input, args...), shape).
/// </summary>
[RuleGenerator]
public sealed partial class CombineActivationsReshape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsCall("call", IsOp<ActivationOp>("activation", op => true), IsVArgsRepeat("parameters", (inputs) =>
        {
            var patterns = new Pattern[inputs.Length];
            patterns[0] = IsReshape(IsWildcard("input"), IsWildcard("shape"));
            for (int i = 1; i < inputs.Length; i++)
            {
                patterns[i] = IsWildcard();
            }

            return patterns;
        }));

    private Expr? GetReplace(ActivationOp activation, Call call, Expr input, IReadOnlyList<Expr> parameters, Expr shape)
    {
        // TODO: Not support PRelu for now.
        if (activation is PRelu)
        {
            return null;
        }

        return Reshape(
            new Call(activation, new Expr[] { input }.Concat(parameters.Skip(1)).ToArray()).InheritMetaData(call),
            shape);
    }
}

/// <summary>
/// reshape(pad(input), shape) => pad(reshape(input, shape)).
/// </summary>
[RuleGenerator]
public sealed partial class CombineReshapePad : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsReshape(
            "reshape",
            "reshapeCall",
            _ => true,
            IsPad("pad", "padCall", _ => true, IsWildcard("input"), IsTensorConst("pads"), IsTensorConst("value")) with { TypePattern = HasFixedShape() },
            IsWildcard("shape")) with
        { TypePattern = HasFixedShape() };

    private Expr? GetReplace(Reshape reshape, Call reshapeCall, Pad pad, Call padCall, Expr input, Expr shape, int[] pads, Expr value)
    {
        // only support pattern like melgan
        var reshapeRank = reshapeCall.CheckedShape.Rank;
        var padRank = padCall.CheckedShape.Rank;
        if (reshapeRank >= padRank
            && Enumerable.SequenceEqual(reshapeCall.CheckedShape.ToValueArray()[(reshapeRank - padRank)..], padCall.CheckedShape.ToValueArray()))
        {
            return Pad(
            Reshape(input, Enumerable.Repeat(1, reshapeRank - padRank).Concat(input.CheckedShape.ToValueArray()).ToArray()).InheritMetaData(reshapeCall),
            Tensor.From(Enumerable.Repeat(0, (reshapeRank - padRank) * 2).Concat(pads).ToArray(), new[] { reshapeRank, 2 }),
            pad.PadMode,
            value).InheritMetaData(padCall);
        }

        return null;
    }
}

/// <summary>
/// combine reshape transpose
/// e.g. :
/// %5 // f32[1,77,768]
/// %6 = Reshape(%5, const(i64[4] : {1L,77L,12L,64L})): // f32[1,77,12,64]
/// %7 = Transpose(%6, const(i64[4] : {0L,2L,1L,3L})): // f32[1,12,77,64]
/// %8 = Reshape(%7, const(i32[3] : {12,77,64})): // f32[12,77,64].
/// after combine :
/// %5 // f32[1,77,768]
/// %6 = Reshape(%5, const(i64[4] : {1L,77L,12L,64L})): // f32[1,77,12,64]
/// %7 = Reshape(%6, const(i64[3] : {77L,12L,64L})): // f32[77L,12L,64L].
/// %8 = Transpose(%7, const(i64[4] : {1L,0L,2L})): // f32[12,77,64].
/// then use foldreshape.
/// </summary>
[RuleGenerator]
public sealed partial class CombineReshapeTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsReshape(
        IsTranspose(
            null,
            "trans",
            IsWildcard("input") with { TypePattern = HasFixedShape() },
            IsTensorConst("perm")) with
        { TypePattern = HasFixedShape() },
        IsTensorConst("newShape"));

    private int FindViewAxis(int[] oldShape, int[] newShape)
    {
        var indices = Enumerable.Range(0, oldShape.Length).ToList();
        foreach (var dim in newShape)
        {
            for (int i = 0; i < oldShape.Length; i++)
            {
                if (oldShape[i] == dim && indices.IndexOf(i) != -1)
                {
                    indices.Remove(i);
                }
            }
        }

        var oneindex = (indices.Count == 1) ? indices[0] : -1;
        return oneindex;
    }

    private Expr? GetReplace(Expr input, Call trans, int[] newShape, int[] perm)
    {
        var transShape = trans.CheckedShape.ToValueArray();

        if (transShape.Length == newShape.Length + 1)
        {
            // check reshape is sequeeze
            var viewAxis = FindViewAxis(transShape, newShape);
            if (viewAxis == -1)
            {
                return null;
            }

            var inv = perm.Select((p, i) => (p, i)).OrderBy(tp => tp.p).ToArray();
            var invViewAxis = inv.Where(tp => tp.i == viewAxis).First().p;
            var invPerm = perm.ToList();
            var invNewShape = input.CheckedShape.ToValueList();
            invNewShape.RemoveAt(invViewAxis);
            invPerm.Remove(invViewAxis);
            return IR.F.Tensors.Transpose(IR.F.Tensors.Reshape(input, invNewShape.ToArray()), invPerm.Select(i => i > invViewAxis ? i - 1 : i).ToArray());
        }
        else if (transShape.Length == newShape.Length - 1)
        {
            // check rehsape is unsequeeze
            return null;
        }

        return null;
    }
}
