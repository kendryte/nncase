// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldLayernormPattern1 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsBinary("add_beta", "add_beta_call", BinaryOp.Add, IsBinary("mul", "mul_call", BinaryOp.Mul, IsReshape("rshape2", "rshape2_call", _ => true, IsBinary("div", "div_call", BinaryOp.Div, IsUnary("sqrt", "sqrt_call", UnaryOp.Sqrt, IsBinary("add_eps", "add_eps_call", BinaryOp.Add, IsReduce("rd2", "rd2_call", ReduceOp.Mean, IsBinary("pow", "pow_call", BinaryOp.Pow, IsBinary("sub", "sub_call", BinaryOp.Sub, IsReduce("rd1", "rd1_call", ReduceOp.Mean, IsReshape("rd1", "rd1_call", _ => true, IsTensorConst("input")))))), IsTensorConst("eps"))))), IsTensorConst("gamma")), IsTensorConst("beta"));

    private Expr? GetReplace(Call add_beta_call, Call mul_call, TensorConst eps, TensorConst gamma, TensorConst beta, TensorConst input)
    {
        var axis = add_beta_call.CheckedShape.Count - mul_call.CheckedShape.Count;
        return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta);
    }
}

[RuleGenerator]
public sealed partial class FoldLayernormPattern2 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsBinary(
        "add_beta", "add_beta_call", BinaryOp.Add, IsBinary("mul", "mul_call", BinaryOp.Mul, IsBinary("div", "div_call", BinaryOp.Div, IsUnary("sqrt", "sqrt_call", UnaryOp.Sqrt, IsBinary("add_eps", "add_eps_call", BinaryOp.Add, IsReduce("rd2", "rd2_call", ReduceOp.Mean, IsBinary("pow", "pow_call", BinaryOp.Pow, IsBinary("sub", "sub_call", BinaryOp.Sub, IsReduce("rd1", "rd1_call", ReduceOp.Mean, IsTensorConst("input"))))), IsTensorConst("eps")))), IsTensorConst("gamma")), IsTensorConst("beta"));

    private Expr? GetReplace(TensorConst add_beta_call, Call mul_call, TensorConst eps, TensorConst gamma, TensorConst beta, TensorConst input)
    {
        var axis = add_beta_call.CheckedShape.Count - mul_call.CheckedShape.Count;
        return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta);
    }
}

[RuleGenerator]
public sealed partial class FoldLayernormPattern3 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "add_all", "add_all_call", BinaryOp.Add, IsBinary("mul_x", "mul_x_call", BinaryOp.Mul, null, IsBinary("mul_gamma", "mul_gamma_call", BinaryOp.Mul, IsUnary("rsqrt", "rsqrt_call", UnaryOp.Rsqrt, IsBinary("add_eps", "add_eps_call", BinaryOp.Add, IsReduce("rd_var", "rd_var_call", ReduceOp.Mean, IsUnary("square", "sqare_call", UnaryOp.Square, IsBinary("sub_mu", "sub_mu_call", BinaryOp.Sub, IsTensorConst(), IsReduce("rd_mu", "rd_mu_call", ReduceOp.Mean, IsTensorConst("input"))))), IsTensorConst("eps"))), IsTensorConst("gamma"))), IsBinary("sub_beta", "sub_beta_call", BinaryOp.Sub, IsTensorConst("beta"), IsBinary("mul_mu", "mul_mu_call", BinaryOp.Mul)));

    private Expr? GetReplace(Call add_beta_call, Call mul_gamma_call, TensorConst eps, TensorConst gamma, TensorConst beta, TensorConst input)
    {
        var axis = add_beta_call.CheckedShape.Count - mul_gamma_call.CheckedShape.Count;
        return LayerNorm(axis, eps.Value.Cast<float>()[0], input, gamma, beta);
    }
}
