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

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Combine Transpose with Binary
/// binary(transpose(a,p),transpose(b,p)) => transpose(binary(a,b),p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineBinaryTranspose : IRewriteRule
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CombineBinaryTranspose"/> class.
    /// </summary>
    public CombineBinaryTranspose()
    {
        var perm = IsWildcard("perm");
        Pattern = IsBinary("binary", "binaryCall", x => true, IsTranspose(IsWildcard("x"), perm), IsTranspose(IsWildcard("y"), perm));
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; init; }

    private Expr? GetReplace(Binary binary, Call binaryCall, Expr x, Expr y, Expr perm)
    {
        return Transpose(Binary(binary.BinaryOp, x, y).InheritMetaData(binaryCall), perm);
    }
}

/// <summary>
/// Combine Transpose with Const Binary, if Const has rank 1.
/// binary(transpose(a,p),const(b)) => transpose(binary(a,const(b)),p) or binary(const(a),transpose(b,p)) => transpose(binary(const(a),b),p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineConstBinaryTranspose : IRewriteRule
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CombineConstBinaryTranspose"/> class.
    /// </summary>
    public CombineConstBinaryTranspose()
    {
        var perm = IsWildcard("perm");
        Pattern = IsAlt(IsBinary("binary", "binaryCall", _ => true, IsTranspose(IsWildcard("x"), perm), IsConst("y") with { TypePattern = HasRank(1) | HasRank(0) }), IsBinary("binary", "binaryCall", _ => true, IsConst("x") with { TypePattern = HasRank(1) | HasRank(0) }, IsTranspose(IsWildcard("y"), perm)));
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; init; }

    private Expr? GetReplace(Binary binary, Call binaryCall, Expr x, Expr y, Expr perm)
    {
        var expandDim = perm.CheckedShape.Size - ((TensorConst)perm).Value.ToArray<int>()[perm.CheckedShape.Size - 1] - 1;

        if (x is Const)
        {
            if (x.CheckedShape.Rank == 0)
            {
                return Transpose(Binary(binary.BinaryOp, x, y).InheritMetaData(binaryCall), perm);
            }

            var newShape = new List<int>() { x.CheckedShape[0].FixedValue };
            if (x.CheckedShape[0].FixedValue != 1)
            {
                for (int i = 0; i < expandDim; i++)
                {
                    newShape.Add(1);
                }
            }

            Expr newConst = Const.FromValue(((Expr)Tensor.From<float>(((TensorConst)x).Value.ToArray<float>(), new Nncase.IR.Shape(newShape))).Evaluate()).InheritMetaData(x);
            return Transpose(Binary(binary.BinaryOp, newConst, y).InheritMetaData(binaryCall), perm);
        }

        if (y is Const)
        {
            if (y.CheckedShape.Rank == 0)
            {
                return Transpose(Binary(binary.BinaryOp, x, y).InheritMetaData(binaryCall), perm);
            }

            var newShape = new List<int>() { y.CheckedShape[0].FixedValue };
            if (y.CheckedShape[0].FixedValue != 1)
            {
                for (int i = 0; i < expandDim; i++)
                {
                    newShape.Add(1);
                }
            }

            var newConst = Const.FromValue(((Expr)Tensor.From<float>(((TensorConst)y).Value.ToArray<float>(), new Nncase.IR.Shape(newShape))).Evaluate()).InheritMetaData(y);
            return Transpose(Binary(binary.BinaryOp, x, newConst).InheritMetaData(binaryCall), perm);
        }

        return null;
    }
}

/// <summary>
/// transpose(binary(x,const),p) => binary(transpose(x,p),new_const).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposeConstBinary : RewriteRule<CallPattern>
{
    public override CallPattern Pattern { get; } =
      IsTranspose(
        IsAlt(
          IsBinary("binary", "binaryCall", _ => true, IsWildcard("x", x => x is not Const), IsTensorConst("y")),
          IsBinary("binary", "binaryCall", _ => true, IsTensorConst("x"), IsWildcard("y", x => x is not Const))),
        IsTensorConst("perm"));

    private Const GetNewConst(TensorConst oldConst, Expr input, TensorConst perm)
    {
        int[] newConstShape;
        if (oldConst.Value.Shape.Rank < input.CheckedShape.Rank)
        {
            newConstShape = Enumerable.Repeat(1, input.CheckedShape.Rank - oldConst.Value.Shape.Rank).Concat(oldConst.Value.Shape.ToValueArray()).ToArray();
        }
        else
        {
            newConstShape = oldConst.Value.Shape.ToValueArray();
        }

        return (Const)Const.FromValue(Transpose(Tensor.FromBytes(oldConst.Value.ElementType, oldConst.Value.BytesBuffer.ToArray(), newConstShape), perm).Evaluate()).InheritMetaData(oldConst);
    }

    private Expr? GetReplace(Binary binary, Call binaryCall, Expr x, Expr y, TensorConst perm)
    {
        if (x is TensorConst && y.CheckedShape.Rank != binaryCall.CheckedShape.Rank)
        {
            return null;
        }

        if (y is TensorConst && x.CheckedShape.Rank != binaryCall.CheckedShape.Rank)
        {
            return null;
        }

        if (x is TensorConst constX)
        {
            return Binary(binary.BinaryOp, GetNewConst(constX, y, perm), Transpose(y, perm)).InheritMetaData(binaryCall);
        }

        var constY = (TensorConst)y;
        return Binary(binary.BinaryOp, Transpose(x, perm), GetNewConst(constY, x, perm)).InheritMetaData(binaryCall);
    }
}

/// <summary>
/// Combine Transpose with Concat
/// concat((transpose(x,p),...), a) => transpose(concat((x,...), p[a]), p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposeConcat : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsConcat(
               "concat",
               "concatCall",
               PatternMatch.Utility.IsTuple(null, IsVArgsRepeat("tupleInputs", exprs =>
               {
                   var patterns = new Pattern[exprs.Length];
                   for (var i = 0; i < patterns.Length; i++)
                   {
                       patterns[i] = IsTranspose(IsWildcard($"input_{i}"), IsTensorConst($"perm_{i}"));
                   }

                   return patterns;
               })),
               IsTensorConst("axis"));

    private Expr? GetReplace(Expr concat, Call concatCall, IReadOnlyList<Expr> tupleInputs, int axis, IMatchResult matchResult)
    {
        var inputs = Enumerable.Range(0, tupleInputs.Count).Select(i => (Expr)matchResult[$"input_{i}"]);
        var perms = new HashSet<Tensor<int>>(Enumerable.Range(0, tupleInputs.Count).Select(i => ((TensorConst)matchResult[$"perm_{i}"]).Value.Cast<int>(CastMode.KDefault)));

        Tensor<int> perm;
        if (perms.Count == 1)
        {
            perm = perms.Single();
        }
        else
        {
            return null;
        }

        return Transpose(Concat(new IR.Tuple(inputs.ToArray()), perm[axis]).InheritMetaData(concatCall), perm);
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
        "padCall",
        x => true,
        IsTranspose(IsWildcard("input"), IsTensorConst("perm")),
        IsWildcard("pads"),
        IsWildcard("padValue"));

    private Expr GetReplace(Pad pad, Call padCall, Expr input, int[] perm, Expr pads, Expr padValue)
    {
        var inv_perm = perm.Select((p, i) => (p, i)).OrderBy(tp => tp.p).ToArray();
        var newPads = new List<Expr>();
        for (var i = 0; i < inv_perm.Length; i++)
        {
            newPads.Add(pads[inv_perm[i].i]);

            // newPads[i] = pads[perm[i]];
        }

        var p = Pad(input, Stack(new IR.Tuple(newPads.ToArray()), 0), pad.PadMode, padValue).InheritMetaData(padCall);
        return Transpose(p, perm);
    }
}

/// <summary>
/// Combine Pad with Transpose
/// transpose(pad(x, pp),p) => pad(transpose(x),new_pp).
/// </summary>
[RuleGenerator]
public sealed partial class CombinePadTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose(
        "transpose",
        x => true,
        IsPad(
            "pad",
            "padCall",
            y => true,
            IsWildcard("input"),
            IsTensorConst("pads"),
            IsTensorConst("padValue")),
        IsTensorConst("perm"));

    private Expr GetReplace(Pad pad, Call padCall, Expr input, int[] perm, Expr pads, Expr padValue)
    {
        var newPads = new List<int>();
        for (int i = 0; i < perm.Length; i++)
        {
            newPads.Add(((TensorConst)pads).Value.ToArray<int>()[perm[i] * 2]);
            newPads.Add(((TensorConst)pads).Value.ToArray<int>()[(perm[i] * 2) + 1]);
        }

        return Pad(Transpose(input, perm), Tensor.From<int>(newPads.ToArray(), pads.CheckedShape), pad.PadMode, padValue).InheritMetaData(padCall);
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
        "reduceCall",
        x => true,
        IsTranspose("tp", "tpCall", _ => true, IsWildcard("input"), IsTensorConst("perm")),
        IsTensorConst("axis"),
        IsWildcard("initValue"),
        IsTensorConst("keepDims", IsBoolScalar()));

    private Expr? GetReplace(Reduce reduce, Call reduceCall, Expr input, Call tpCall, int[] perm, int[] axis, Expr initValue, bool keepDims)
    {
        // var newAxis = Gather(perm, 0, axis);
        // var tp = Transpose(Reduce(reduce.ReduceOp, input, newAxis, initValue, true), perm);
        // return keepDims ? tp : Squeeze(tp, axis);
        var newAxis = new List<int>();
        for (int i = 0; i < axis.Length; i++)
        {
            newAxis.Add(perm[axis[i] >= 0 ? axis[i] : axis[i] + tpCall.CheckedShape.Rank]);
        }

        var newPerm = new List<int>();
        for (int i = 0; i < perm.Length; i++)
        {
            newPerm.Add(perm[i]);
        }

        if (!keepDims)
        {
            var sortedNewAxis = newAxis;
            sortedNewAxis.Sort((a, b) => b.CompareTo(a));
            for (int i = 0; i < sortedNewAxis.Count; i++)
            {
                var it = newPerm.Find(x => x == sortedNewAxis[i]);
                newPerm.Remove(it);
                for (int j = 0; j < newPerm.Count; j++)
                {
                    newPerm[j] = newPerm[j] > sortedNewAxis[i] ? newPerm[j] - 1 : newPerm[j];
                }
            }
        }

        return Transpose(Reduce(reduce.ReduceOp, input, newAxis.ToArray(), initValue, keepDims).InheritMetaData(reduceCall), newPerm.ToArray());
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
    public IPattern Pattern { get; } = IsUnary("unary", "unaryCall", x => true, IsTranspose(IsWildcard("input"), IsWildcard("perm")));

    private Expr? GetReplace(Unary unary, Call unaryCall, Expr input, Expr perm)
    {
        return Transpose(Unary(unary.UnaryOp, input).InheritMetaData(unaryCall), perm);
    }
}

/// <summary>
/// transpose(activation(x),perm) => activation(transpose(x,perm)).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposeActivations : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsTranspose(
            IsCall("actCall", IsOp<ActivationOp>("activation", op => true), IsVArgsRepeat("parameters", () => IsWildcard())),
            IsWildcard("perm"));

    private Expr GetReplace(Call actCall, ActivationOp activation, IReadOnlyList<Expr> parameters, Expr perm)
    {
        var newcall = new Call(
            activation,
            new Expr[] { Transpose(parameters[0], perm) }
                .Concat(parameters.Skip(1)).ToArray());
        newcall.InheritMetaData(actCall);
        return newcall;
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
      IsCall("actCall", IsOp<ActivationOp>("activation", op => true), IsVArgsRepeat("parameters", (inputs) =>
      {
          var patterns = new Pattern[inputs.Length];
          patterns[0] = IsTranspose(IsWildcard("input"), IsWildcard("perm"));
          for (int i = 1; i < inputs.Length; i++)
          {
              patterns[i] = IsWildcard();
          }

          return patterns;
      }));

    private Expr? GetReplace(Call actCall, ActivationOp activation, Expr input, IReadOnlyList<Expr> parameters, Expr perm)
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
                return Transpose(new Call(activation, input, new_slope), perm);
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

            return Transpose(new Call(activation, input, new_slope), perm);
        }

        var newCall = new Call(activation, new Expr[] { input }.Concat(parameters.Skip(1)).ToArray());
        newCall.InheritMetaData(actCall);
        return Transpose(
          newCall,
          perm);
    }
}
