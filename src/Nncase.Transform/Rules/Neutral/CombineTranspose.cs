// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
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

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Combine Transpose with Binary
/// binary(transpose(a,p),transpose(b,p)) => transpose(binary(a,b),p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposeBinary : IRewriteRule
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CombineTransposeBinary"/> class.
    /// </summary>
    public CombineTransposeBinary()
    {
        var perm = IsWildcard("perm");
        Pattern = IsBinary("binary", x => true, IsTranspose(IsWildcard("x"), perm), IsTranspose(IsWildcard("y"), perm));
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; init; }

    private Expr? GetReplace(Binary binary, Expr x, Expr y, Expr perm)
    {
        return Transpose(Binary(binary.BinaryOp, x, y), perm);
    }
}

/// <summary>
/// Combine Transpose with Concat
/// concat((transpose(x,p),...), a) => transpose(concat((x,...), p[a]), p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposeConcat : IRewriteRule
{
    private ExprPattern[]? _inputsPat;

    /// <summary>
    /// Initializes a new instance of the <see cref="CombineTransposeConcat"/> class.
    /// </summary>
    public CombineTransposeConcat()
    {
        var perm = IsWildcard("perm");
        Pattern = IsConcat(
            IsTuple(IsVArgsRepeat(exprs =>
            {
                _inputsPat = new ExprPattern[exprs.Count];
                var patterns = new Pattern[exprs.Count];

                for (var i = 0; i < _inputsPat.Length; i++)
                {
                    var input = IsWildcard();
                    _inputsPat[i] = input;
                    patterns[i] = IsTranspose(input, perm);
                }

                return patterns;
            })),
            IsWildcard("axis"));
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; }

    private Expr? GetReplace(IMatchResult mr, Expr perm, Expr axis)
    {
        var inputs = _inputsPat!.Select(x => mr.Get(x)).ToArray();
        return Transpose(Concat(new IR.Tuple(inputs), perm[axis]), perm);
    }
}

/// <summary>
/// Combine Transpose with Pad
/// pad(transpose(x,p), pp) => transpose(pad(x, invtranspose(pp, p)), p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposePad : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsPad(
        "pad",
        x => true,
        IsTranspose(IsWildcard("input"), IsTensorConst("perm")),
        IsWildcard("pads"),
        IsWildcard("padValue"));

    private Expr GetReplace(Pad pad, Expr input, int[] perm, Expr pads, Expr padValue)
    {
        var inv_perm = perm.Select((p, i) => (p, i)).OrderBy(tp => tp.p).ToArray();
        var newPads = new List<Expr>();
        for (var i = 0; i < inv_perm.Length; i++)
        {
            newPads.Add(pads[inv_perm[i].i]);
            // newPads[i] = pads[perm[i]];
        }

        var p = Pad(input, Stack(new IR.Tuple(newPads), 0), pad.PadMode, padValue);
        return Transpose(p, perm);
    }
}

/// <summary>
/// Combine Transpose with Reduce
/// reduce(transpose(x,p), a) => transpose(reduce(x, gather(p, a)), p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposeReduce : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsReduce(
        "reduce",
        x => true,
        IsTranspose(IsWildcard("input"), IsWildcard("perm")),
        IsWildcard("axis"),
        IsWildcard("initValue"),
        IsTensorConst("keepDims", IsBoolScalar()));

    private Expr? GetReplace(Reduce reduce, Expr input, Expr perm, Expr axis, Expr initValue, bool keepDims)
    {
        var newAxis = Gather(perm, 0, axis);
        var tp = Transpose(Reduce(reduce.ReduceOp, input, newAxis, initValue, true), perm);
        return keepDims ? tp : Squeeze(tp, axis);
    }
}

/// <summary>
/// Combine Transpose with Unary
/// reduce(transpose(x,p), a) => transpose(reduce(x, invtranspose(a, p)), p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposeUnary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsUnary("unary", x => true, IsTranspose(IsWildcard("input"), IsWildcard("perm")));

    private Expr? GetReplace(Unary unary, Expr input, Expr perm)
    {
        return Transpose(Unary(unary.UnaryOp, input), perm);
    }
}


/// <summary>
/// transpose(activation(x),perm) => activation(transpose(x,perm))
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposeActivations : IRewriteRule
{
    public IPattern Pattern { get; } =
      IsTranspose(
       IsCall(IsOp<ActivationOp>("activation", op => true), IsVArgsRepeat("parameters", () => IsWildcard())),
       IsWildcard("perm")
      );

    private Expr GetReplace(ActivationOp activation, IReadOnlyList<Expr> parameters, Expr perm)
    {
        return new Call(activation,
          new Expr[] { Transpose(parameters[0], perm) }
            .Concat(parameters.Skip(1)).ToArray());
    }
}