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
using Nncase.PatternMatch;
using Tensorflow;
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
        Pattern = IsBinary("binary", x => true, IsReshape(IsWildcard("x"), shape), IsReshape(IsWildcard("y"), shape));
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; init; }

    private Expr? GetReplace(Binary binary, Expr x, Expr y, Expr shape)
    {
        if (x.CheckedShape == y.CheckedShape)
        {
            return Reshape(Binary(binary.BinaryOp, x, y), shape);
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

        if (constInput.CheckedShape.Rank == 0)
        {
            var res = Reshape(Binary(binary.BinaryOp, input, constInput), shape);
            res.InferenceType();
            return res;
        }

        var significantInputShape = input.CheckedShape.ToValueArray().Where(x => x > 1).ToArray();
        bool leftConst = ReferenceEquals(callParams[0], constInput);
        var constSize = constInput.CheckedShape.ToValueArray()[0];
        if (significantShape.SequenceEqual(significantInputShape) && oldShape[^1] == constSize)
        {
            var broadcastIndex = Array.LastIndexOf(input.CheckedShape.ToValueArray(), constSize);
            var newConstShape = Enumerable.Repeat(1, input.CheckedShape.Rank - 1 - broadcastIndex).ToList();
            newConstShape.Insert(0, constSize);

            var res = Reshape(Binary(binary.BinaryOp, leftConst ? Reshape(constInput, newConstShape.ToArray()) : input, leftConst ? input : Reshape(constInput, newConstShape.ToArray())), call.CheckedShape);
            res.InferenceType();
            return res;
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

    private Expr? GetReplace(Unary unary, Expr input, Expr shape)
    {
        return Reshape(
            Unary(unary.UnaryOp, input),
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
        IsCall(IsOp<ActivationOp>("activation", op => true), IsVArgsRepeat("parameters", (inputs) =>
        {
            var patterns = new Pattern[inputs.Length];
            patterns[0] = IsReshape(IsWildcard("input"), IsWildcard("shape"));
            for (int i = 1; i < inputs.Length; i++)
            {
                patterns[i] = IsWildcard();
            }

            return patterns;
        }));

    private Expr? GetReplace(ActivationOp activation, Expr input, IReadOnlyList<Expr> parameters, Expr shape)
    {
        // TODO: Not support PRelu for now.
        if (activation is PRelu)
        {
            return null;
        }

        return Reshape(
            new Call(activation, new Expr[] { input }.Concat(parameters.Skip(1)).ToArray()),
            shape);
    }
}
