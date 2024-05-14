// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using DryIoc;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldLayerNormPattern1 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "addBeta",
            "addBetaCall",
            BinaryOp.Add,
            IsBinary(
                "mul",
                "mulCall",
                BinaryOp.Mul,
                IsReshape(
                    "rshape2",
                    "rshape2Call",
                    _ => true,
                    IsBinary(
                        "div",
                        "divCall",
                        BinaryOp.Div,
                        IsWildcard(),
                        IsUnary(
                            "sqrt",
                            "sqrtCall",
                            UnaryOp.Sqrt,
                            IsBinary(
                                "addEps",
                                "addEpsCall",
                                BinaryOp.Add,
                                IsReduce(
                                    "rd2",
                                    "rd2Call",
                                    ReduceOp.Mean,
                                    IsBinary(
                                        "pow",
                                        "powCall",
                                        BinaryOp.Pow,
                                        IsBinary(
                                            "sub",
                                            "subCall",
                                            BinaryOp.Sub,
                                            IsWildcard(),
                                            IsReduce(
                                                "rd1",
                                                "rd1Call",
                                                ReduceOp.Mean,
                                                IsReshape(
                                                    "reshape1",
                                                    "reshape1Call",
                                                    _ => true,
                                                    IsWildcard("input")))))),
                                IsTensorConst("eps"))))),
                IsTensorConst("gamma")),
            IsTensorConst("beta"));

    private Expr? GetReplace(Call addBetaCall, Call subCall, Call rd1Call, Call divCall, TensorConst eps, TensorConst gamma, TensorConst beta, Expr input)
    {
        if (subCall[Binary.Lhs] == rd1Call[Reduce.Input] && divCall[Binary.Lhs] == subCall)
        {
            var axis = addBetaCall.CheckedShape.Count - gamma.CheckedShape.Count;
            bool cFirst = false;
            var axes = rd1Call[Reduce.Axis].Evaluate().AsTensor().ToArray<int>();
            if (axes.Length == 1 && axes[0] != input.CheckedShape.Count - 1 && axes[0] != -1)
            {
                cFirst = true;
            }

            return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta, channelFirst: cFirst);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldLayerNormPattern2 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "addBeta",
            "addBetaCall",
            BinaryOp.Add,
            IsBinary(
            "mul",
            "mulCall",
            BinaryOp.Mul,
            IsBinary(
                "div",
                "divCall",
                BinaryOp.Div,
                IsWildcard(),
                IsUnary(
                    "sqrt",
                    "sqrtCall",
                    UnaryOp.Sqrt,
                    IsBinary(
                        "addEps",
                        "addEpsCall",
                        BinaryOp.Add,
                        IsReduce(
                            "rd2",
                            "rd2Call",
                            ReduceOp.Mean,
                            IsBinary(
                                "pow",
                                "powCall",
                                BinaryOp.Pow,
                                IsBinary(
                                    "sub",
                                    "subCall",
                                    BinaryOp.Sub,
                                    IsWildcard(),
                                    IsReduce(
                                        "rd1",
                                        "rd1Call",
                                        ReduceOp.Mean,
                                        IsWildcard("input"))))),
                        IsTensorConst("eps")))),
            IsTensorConst("gamma")),
            IsTensorConst("beta"));

    private Expr? GetReplace(Expr addBetaCall, Call subCall, Call rd1Call, Call divCall, TensorConst eps, TensorConst gamma, TensorConst beta, Expr input)
    {
        if ((subCall[Binary.Lhs] == rd1Call[Reduce.Input] || subCall[Binary.Rhs] == rd1Call[Reduce.Input]) &&
            divCall[Binary.Lhs] == subCall)
        {
            var axis = addBetaCall.CheckedShape.Count - gamma.CheckedShape.Count;
            bool cFirst = false;
            var axes = rd1Call[Reduce.Axis].Evaluate().AsTensor().ToArray<int>();
            if (axes.Length == 1 && axes[0] != input.CheckedShape.Count - 1 && axes[0] != -1)
            {
                cFirst = true;
            }

            return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta, channelFirst: cFirst);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldLayerNormPattern3 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "addAll",
            "addAllCall",
            BinaryOp.Add,
            IsBinary(
                "mulX",
                "mulXCall",
                BinaryOp.Mul,
                IsWildcard(),
                IsBinary(
                    "mulGamma",
                    "mulGammaCall",
                    BinaryOp.Mul,
                    IsUnary(
                        "rsqrt",
                        "rsqrtCall",
                        UnaryOp.Rsqrt,
                        IsBinary(
                            "addEps",
                            "addEpsCall",
                            BinaryOp.Add,
                            IsReduce(
                                "rdVar",
                                "rdVarCall",
                                ReduceOp.Mean,
                                IsUnary(
                                    "square",
                                    "sqareCall",
                                    UnaryOp.Square,
                                    IsBinary(
                                        "subMu",
                                        "subMuCall",
                                        BinaryOp.Sub,
                                        IsWildcard(),
                                        IsReduce(
                                            "rdMu",
                                            "rdMuCall",
                                            ReduceOp.Mean,
                                            IsWildcard("input"))))),
                            IsTensorConst("eps"))),
                    IsTensorConst("gamma"))),
            IsBinary(
                "subBeta",
                "subBetaCall",
                BinaryOp.Sub,
                IsTensorConst("beta"),
                IsBinary(
                    "mulMu",
                    "mulMuCall",
                    BinaryOp.Mul)));

    private Expr? GetReplace(Call addAllCall, Call mulMuCall, Call subMuCall, Call mulXCall, Call rdMuCall, TensorConst eps, TensorConst gamma, TensorConst beta, Expr input)
    {
        if (mulMuCall[Binary.Lhs] == subMuCall[Binary.Rhs] && mulMuCall[Binary.Rhs] == mulXCall[Binary.Rhs] &&
            mulXCall[Binary.Lhs] == subMuCall[Binary.Lhs] && mulXCall[Binary.Lhs] == rdMuCall[Reduce.Input])
        {
            var axis = addAllCall.CheckedShape.Count - gamma.CheckedShape.Count;
            bool cFirst = false;
            var axes = rdMuCall[Reduce.Axis].Evaluate().AsTensor().ToArray<int>();
            if (axes.Length == 1 && axes[0] != input.CheckedShape.Count - 1 && axes[0] != -1)
            {
                cFirst = true;
            }

            return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta, channelFirst: cFirst);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldLayerNormPattern4 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "addAll",
            "addAllCall",
            BinaryOp.Add,
            IsBinary(
                "mulX",
                "mulXCall",
                BinaryOp.Mul,
                IsWildcard("input"),
                IsBinary(
                    "mulGamma",
                    "mulGammaCall",
                    BinaryOp.Mul,
                    IsUnary(
                        "rsqrt",
                        "rsqrtCall",
                        UnaryOp.Rsqrt,
                        IsBinary(
                            "addEps",
                            "addEpsCall",
                            BinaryOp.Add,
                            IsReduce(
                                "rdVar",
                                "rdVarCall",
                                ReduceOp.Mean,
                                IsBinary(
                                        "subMul",
                                        "subMulCall",
                                        BinaryOp.Mul,
                                        IsBinary(
                                            "sub",
                                            "subCall",
                                            BinaryOp.Sub,
                                            IsWildcard(),
                                            IsReduce(
                                                "mean",
                                                "meanCall",
                                                ReduceOp.Mean)),
                                        IsWildcard())),
                            IsTensorConst("eps"))),
                    IsTensorConst("gamma"))),
            IsBinary(
                "subBeta",
                "subBetaCall",
                BinaryOp.Sub,
                IsTensorConst("beta"),
                IsBinary(
                    "mulMu",
                    "mulMuCall",
                    BinaryOp.Mul,
                    IsWildcard(),
                    IsWildcard())));

    private Expr? GetReplace(Call addAllCall, Call subCall, Call mulXCall, Call mulMuCall, Call subMulCall, Call meanCall, TensorConst eps, TensorConst gamma, TensorConst beta, Expr input)
    {
        if (subMulCall[Binary.Lhs] == subMulCall[Binary.Rhs] && mulXCall[Binary.Rhs] == mulMuCall[Binary.Rhs] &&
            subCall[Binary.Lhs] == mulXCall[Binary.Lhs] && subCall[Binary.Rhs] == mulMuCall[Binary.Lhs] && mulXCall[Binary.Lhs] == meanCall[Reduce.Input])
        {
            var axis = addAllCall.CheckedShape.Count - gamma.CheckedShape.Count;
            bool cFirst = false;
            var axes = meanCall[Reduce.Axis].Evaluate().AsTensor().ToArray<int>();
            if (axes.Length == 1 && axes[0] != input.CheckedShape.Count - 1 && axes[0] != -1)
            {
                cFirst = true;
            }

            return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta, channelFirst: cFirst);
        }

        return null;
    }
}

// pattern from llama without mean and beta
[RuleGenerator]
public sealed partial class FoldLayerNormPattern5 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "mulGamma",
            "mulGammaCall",
            BinaryOp.Mul,
            IsTensorConst("gamma"),
            IsBinary(
                "mulX",
                "mulXCall",
                BinaryOp.Mul,
                IsWildcard("input"),
                IsBinary(
                    "rsqrt",
                    "rsqrtCall",
                    BinaryOp.Div,
                    IsTensorConst("one"),
                    IsUnary(
                        "sqrt",
                        "sqrtCall",
                        UnaryOp.Sqrt,
                        IsBinary(
                            "addEps",
                            "addEpsCall",
                            BinaryOp.Add,
                            IsReduce(
                                "rdVar",
                                "rdVarCall",
                                ReduceOp.Mean,
                                IsBinary(
                                        "pow2",
                                        "pow2Call",
                                        BinaryOp.Pow,
                                        IsWildcard(),
                                        IsTensorConst("two"))),
                            IsTensorConst("eps"))))));

    private Expr? GetReplace(Call pow2Call, Call rdVarCall, TensorConst eps, TensorConst gamma, Expr input, TensorConst one, TensorConst two)
    {
        if (input == pow2Call[Binary.Lhs] && one.Value.Cast<float>()[0] == 1f && two.Value.Cast<float>()[0] == 2f)
        {
            var axis = pow2Call.CheckedShape.Count - gamma.CheckedShape.Count;
            var beta = Tensor.FromScalar(0f, gamma.CheckedShape);
            bool cFirst = false;
            var axes = rdVarCall[Reduce.Axis].Evaluate().AsTensor().ToArray<int>();
            if (axes.Length == 1 && axes[0] != input.CheckedShape.Count - 1 && axes[0] != -1)
            {
                cFirst = true;
            }

            return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta, hasMean: false, channelFirst: cFirst);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class ConvertLayerNormChannelFirstToLast : RewriteRule<CallPattern>
{
    public override CallPattern Pattern { get; } =
        IsLayerNorm(
            "ln",
            "_",
            _ => true,
            IsWildcard("x"),
            IsWildcard("scale"),
            IsWildcard("bias"));

    private static List<int> GetPermWithAxis(long axis, int shapeSize)
    {
        var perm = new List<int>();
        for (int i = 0; i < shapeSize; i++)
        {
            if (i != axis)
            {
                perm.Add(i);
            }
        }

        perm.Add((int)axis);
        return perm;
    }

    private Expr? GetReplace(LayerNorm ln, Expr x, Expr scale, Expr bias)
    {
        if (!ln.ChannelFirst)
        {
            return null;
        }

        int axis = ln.Axis;
        float eps = ln.Epsilon;
        bool useMean = ln.UseMean;
        if ((axis == x.CheckedShape.Count - 1) || (axis == -1))
        {
            return null;
        }

        var inPerm = GetPermWithAxis(axis, x.CheckedShape.Count);
        var outPerm = new List<int>();
        for (int i = 0; i < inPerm.Count; i++)
        {
            outPerm.Add(inPerm[inPerm[i]]);
        }

        var newScale = scale;
        var newBias = bias;

        // the permutation of scale and bias must be the same.
        if (scale.CheckedShape.Count != 1 && bias.CheckedShape.Count != 1)
        {
            int axisGap = x.CheckedShape.Count - scale.CheckedShape.Count;
            if (axisGap > axis)
            {
                // Never reach here.
                return null;
            }

            var constPerm = GetPermWithAxis(axis - axisGap, scale.CheckedShape.Count);

            newScale = Tensors.Transpose(scale, constPerm.ToArray());
            newBias = Tensors.Transpose(bias, constPerm.ToArray());
        }

        return Tensors.Transpose(LayerNorm(x.CheckedShape.Count - 1, eps, Tensors.Transpose(x, inPerm.ToArray()), newScale, newBias, useMean, true), outPerm.ToArray());
    }
}
