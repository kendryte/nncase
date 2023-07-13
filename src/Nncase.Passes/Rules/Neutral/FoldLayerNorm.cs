// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.PatternMatch.F.Math;
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
            return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta);
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
            return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta);
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
            return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta);
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
            return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta);
        }

        return null;
    }
}
