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
using Nncase.Utilities;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;
using Tuple = System.Tuple;

namespace Nncase.Passes.Rules.WithMarker;

/// <summary>
/// transpose(activation(x),perm) => activation(transpose(x,perm)).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposeActivations : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        HasMarker(
            IsTranspose(
            HasMarker(
                IsCall("actCall", IsOp<ActivationOp>("activation", op => true), IsVArgsRepeat("arguments", () => IsWildcard() with { TypePattern = HasFixedShape() })),
                "outputMarker"),
            IsTensorConst("perm")),
            "transposeMarker");

    private Expr? GetReplace(Call actCall, ActivationOp activation, IReadOnlyList<Expr> arguments, int[] perm, Marker transposeMarker, Marker outputMarker)
    {
        // todo: argument 1 is marker
        var newArgs = new List<Expr>();
        foreach (var arg in arguments)
        {
            if (arg.CheckedShape.IsScalar)
            {
                newArgs.Add(arg);
                continue;
            }
            else if (arg.CheckedShape.Rank <= perm.Length)
            {
                newArgs.Add(transposeMarker.With(target: Transpose(arg, perm.Select(p => p - (perm.Length - arg.CheckedShape.Rank)).Where(p => p >= 0).ToArray())));
                continue;
            }
            else
            {
                return null;
            }
        }

        var newcall = new Call(activation, newArgs.ToArray());
        newcall.InheritMetaData(actCall);
        return outputMarker.With(target: newcall);
    }
}

/// <summary>
/// activations(transpose(input,p),args...) => transpose(activations(input,args...),p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineActivationsTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
      HasMarker(
          IsCall("actCall", IsOp<ActivationOp>("activation", op => true), IsVArgsRepeat("parameters", (inputs) =>
          {
              var patterns = new Pattern[inputs.Length];
              patterns[0] = HasMarker(
                  IsTranspose(IsWildcard("input"), IsWildcard("perm")),
                  "inputMarker");
              for (int i = 1; i < inputs.Length; i++)
              {
                  patterns[i] = IsWildcard();
              }

              return patterns;
          })),
          "outputMarker");

    private Expr? GetReplace(Call actCall, ActivationOp activation, Expr input, IReadOnlyList<Expr> parameters, Expr perm, Marker inputMarker, Marker outputMarker)
    {
        // note the prelu scope can be broadcast with inputs.
        if (activation is PRelu && parameters[1].CheckedShape.Rank > 1)
        {
            if (perm is not TensorConst const_perm || parameters[1] is not TensorConst slope)
            {
                return null;
            }

            // eg. transpose(input,perm) shape = [1,32,32,8], scope = [1,1,8]
            Expr new_slope;
            var perms = const_perm.Value.ToArray<int>();
            if (slope.Value.Shape.Rank == input.CheckedShape.Rank - 1)
            {
                if (perms[0] != 0)
                {
                    return null;
                }

                var inv_perm = perms.Skip(1).Select((p, i) => (p - 1, i)).OrderBy(tp => tp.Item1).Select(tp => tp.i).ToArray();
                new_slope = Const.FromValue(Transpose(slope, inv_perm).Evaluate());
                return outputMarker.With(target: Transpose(outputMarker.With(target: new Call(activation, inputMarker.With(target: input), new_slope)), perm));
            }
            else if (slope.Value.Shape.Rank == input.CheckedShape.Rank)
            {
                var inv_perm = perms.Select((p, i) => (p, i)).OrderBy(tp => tp.p).Select(tp => tp.i).ToArray();
                new_slope = Const.FromValue(Transpose(slope, inv_perm).Evaluate());
            }
            else
            {
                return null;
            }

            return outputMarker.With(target: Transpose(outputMarker.With(target: new Call(activation, inputMarker.With(target: input), new_slope)), perm));
        }

        var newCall = new Call(activation, new Expr[] { input }.Concat(parameters.Skip(1)).ToArray());
        newCall.InheritMetaData(actCall);
        return outputMarker.With(target: Transpose(
          outputMarker.With(target: newCall),
          perm));
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
        HasMarker(
            IsCall("call", IsOp<ActivationOp>("activation", op => true), IsVArgsRepeat("parameters", (inputs) =>
        {
            var patterns = new Pattern[inputs.Length];
            patterns[0] = HasMarker(IsReshape(IsWildcard("input"), IsWildcard("shape")), "inputMarker");
            for (int i = 1; i < inputs.Length; i++)
            {
                patterns[i] = IsWildcard();
            }

            return patterns;
        })),
            "outMarker");

    private Expr? GetReplace(ActivationOp activation, Call call, Expr input, IReadOnlyList<Expr> parameters, Expr shape, Marker inputMarker, Marker outMarker)
    {
        // TODO: Not support PRelu for now.
        if (activation is PRelu)
        {
            return null;
        }

        return outMarker.With(target: Reshape(
            new Call(activation, new Expr[] { inputMarker.With(target: input) }.Concat(parameters.Skip(1)).ToArray()).InheritMetaData(call),
            shape));
    }
}

[RuleGenerator]
public partial class FoldTransposeActTranspose : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsTranspose(
        "outTr",
        "outTrCall",
        LeakyReluPattern,
        IsWildcard("perm2"));

    public Pattern LeakyReluPattern => HasMarker(
        IsLeakyRelu(
            "op",
            "call",
            HasMarker(
                IsTranspose(IsWildcard("input") with { TypePattern = HasFixedShape() }, IsWildcard("perm1")),
                "inMarker"),
            IsWildcard("alpha")),
        "outMarker");

    private Expr? GetReplace(Call call, Expr input, Marker inMarker, Marker outMarker, int[] perm1, int[] perm2, Call outTrCall, Expr alpha)
    {
        if (perm1.Length != perm2.Length)
        {
            return null;
        }

        if (outTrCall.CheckedShape.SequenceEqual(input.CheckedShape))
        {
            return outMarker.With(target: ReplaceUtility.ReplaceCallFirstParam(call, inMarker.With(target: input), call));
        }

        // transpose(leakyrelu(transpose(input))) => leakyRelu(transpose(transpose(input)))
        else
        {
            return outMarker.With(target: Transpose(outMarker.With(target: Transpose(outMarker.With(target: LeakyRelu(inMarker.With(target: input), alpha)), perm1)), perm2));
        }
    }
}

[RuleGenerator]
public partial class FoldTransposeBinaryActTranspose : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsTranspose(
        HasMarker(
            IsReshape(
                    HasMarker(
                    IsLeakyRelu(
                        "op",
                        "call",
                        HasMarker(
                            IsBinary(
                                "bn",
                                "bnCall",
                                BinaryOp.Add,
                                HasMarker(
                                    IsReshape(
                                        HasMarker(
                                            IsTranspose(HasMarker(IsWildcard(), "input"), IsWildcard("perm1"))),
                                        IsWildcard())),
                                IsWildcard("rhs")),
                            "bnMarker"),
                        IsWildcard("alpha"))),
                    IsWildcard()),
            "outMarker"),
        IsWildcard("perm2"));

    private Expr? GetReplace(int[] perm1, int[] perm2, Expr input, Expr rhs, Marker bnMarker, Marker outMarker, Expr alpha)
    {
        if (perm1.SequenceEqual(new[] { 0, 2, 3, 1 }) && perm2.SequenceEqual(new[] { 0, 3, 1, 2 }))
        {
            // transpose shape check
            // input no marker
            if (rhs is Marker m)
            {
                var constRhs = m.With(target: Reshape(m.Target, new[] { rhs.CheckedShape.Size, 1, 1 }).Evaluate().AsTensor());
                return outMarker.With(target: LeakyRelu(bnMarker.With(target: Add(input, constRhs)), alpha));
            }

            return outMarker.With(target: LeakyRelu(bnMarker.With(target: Add(input, Reshape(rhs, new[] { rhs.CheckedShape.Size, 1, 1 }))), alpha));
        }

        return null;
    }
}
