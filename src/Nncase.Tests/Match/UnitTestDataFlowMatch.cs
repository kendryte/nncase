// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Nncase;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Mutators;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.MatchTest;

[AutoSetupTestMethod(InitSession = false)]
public class UnitTestDataFlowMatch : TestClassBase
{
    [Fact]
    public void TestMatchDataFlowCallCommutive()
    {
        Var x = "x", y = "y";
        var addpat = IsBinary(BinaryOp.Add, IsVar(), IsVar());
        Assert.True(CompilerServices.TryMatchRoot(x + y, addpat, out var _));
        Assert.True(CompilerServices.TryMatchRoot(y + x, addpat, out var _));
        var mulpat = IsBinary(BinaryOp.Mul, IsVar(), IsVar());
        Assert.True(CompilerServices.TryMatchRoot(y * x, mulpat, out var _));
        Assert.True(CompilerServices.TryMatchRoot(x * y, mulpat, out var _));
    }

    [Fact]
    public void TestMatchDataFlowNoCallCommutive()
    {
        Var x = "x", y = "y";
        var subpat = IsBinary(BinaryOp.Sub, x, y);
        Assert.True(CompilerServices.TryMatchRoot(x - y, subpat, out var _));
        Assert.True(CompilerServices.TryMatchRoot(y - x, subpat, out var _));
        var mulpat = IsBinary(BinaryOp.Div, x, y);
        Assert.True(CompilerServices.TryMatchRoot(x / y, mulpat, out var _));
        Assert.True(CompilerServices.TryMatchRoot(y / x, mulpat, out var _));
    }

    [Fact]
    public void TestMatchDataFlowCall()
    {
        Var x = "x", y = "y";
        var addpat = IsBinary(BinaryOp.Add, IsWildcard(), IsWildcard());
        Assert.True(CompilerServices.TryMatchRoot(x + y, addpat, out var _));

        var callpat = IsWildcard();
        Assert.True(CompilerServices.TryMatchRoot(Square(x), callpat, out var _));
        Assert.True(CompilerServices.TryMatchRoot(x + y, callpat, out var _));
    }

    [Fact]
    public void TestNoMatchDataFlowFunc()
    {
        Var x = "x", y = "y";
        var pat = IsBinary(BinaryOp.Add, IsWildcard(), IsWildcard());
        Assert.False(CompilerServices.TryMatchRoot(x - y, pat, out var _));
    }

    [Fact]
    public void TestMatchDataFlowConst()
    {
        Var x = "x", y = "y";
        var pat = IsBinary(BinaryOp.Sub, IsWildcard(), IsConst());
        Assert.True(CompilerServices.TryMatchRoot(x + y - 100, pat, out var _));
    }

    [Fact]
    public void TestMatchDataFlowTuple()
    {
        Var x = "x", y = "y";
        var z = x + y;
        var tuple = new IR.Tuple(x, y, z);
        var tuplepat = PatternMatch.Utility.IsTuple("tp", new Pattern[] { IsVar(), IsWildcard(), IsBinary(BinaryOp.Add, IsWildcard(), IsWildcard()) });

        Assert.True(CompilerServices.TryMatchRoot(tuple, tuplepat, out var _));

        var tuplepat2 = PatternMatch.Utility.IsTuple("tp");
        Assert.True(CompilerServices.TryMatchRoot(tuple, tuplepat2, out var _));
    }

    [Fact]
    [AutoSetupTestMethod(InitSession = true)]
    public void TestNotMatchFoldConstCall()
    {
        var rule = new Passes.Rules.Neutral.FoldConstCall();
        Var x = "x";
        var z = x + 1;
        Assert.False(CompilerServices.TryMatchRoot(z, rule.Pattern, out var _));
    }

    [Fact]
    [AutoSetupTestMethod(InitSession = true)]
    public void TestMatchFoldConstCallTupleWithConst()
    {
        var rule = new Passes.Rules.Neutral.FoldConstCall();

        var z = Concat(new IR.Tuple((Const)2, (Const)1, (Const)2), 0);
        CompilerServices.InferenceType(z);
        Assert.True(CompilerServices.TryMatchRoot(z, rule.Pattern, out var _));
    }

    [Fact]
    [AutoSetupTestMethod(InitSession = true)]
    public void TestMatchFoldConstCallTwiceFalse()
    {
        var rule = new Passes.Rules.Neutral.FoldConstCall();

        var z = Concat(new IR.Tuple(new Var("x", TensorType.Scalar(DataTypes.Int32)), 1, 2), 0);
        CompilerServices.InferenceType(z);
        Assert.False(CompilerServices.TryMatchRoot(z, rule.Pattern, out var _));

        var z1 = Concat(new IR.Tuple(4, 1, 1, 2), 0);
        CompilerServices.InferenceType(z1);
        Assert.True(CompilerServices.TryMatchRoot(z1, rule.Pattern, out var _));
    }

    [Fact]
    public void TestMatchSameInput()
    {
        var input_pat = IsWildcard("input");
        var pat = IsBinary(BinaryOp.Mul, input_pat, IsSigmoid(input_pat));

        var input = (Expr)new[] { 1, 2, 3, 4 };
        var expr = input * IR.F.NN.Sigmoid(input);
        Assert.True(CompilerServices.TryMatch(expr, pat, out var res));
        Assert.NotNull(res["input"]);
    }

    [Fact]
    [AutoSetupTestMethod(InitSession = true)]
    public void TestMatchMultiBranch()
    {
        /*
        - [ ] dataflow rewrite的pr有bug, 这里只考虑了被match到的节点修改后添加的rewritememo中
            1. 但是实际上如果 ((a - (b + c)) + (c * d)) 修改了b + c => b - c, 然后 b + c添加到rewrite memo中了,但是后续的a - (b+c)是mutator自动构造出来的, 此时在遍历老的expr的时候还是看到是a - (b+c), 并没有起到rewrite memo的作用
            2. 我理解应该是类似egraph, 如果遍历的是被替换过的新节点, 那么之间visit 新节点, 否则应该match老节点的类型, 然后leaf节点则匹配自动update 的ExprMemo中的节点.
         */
        Expr a = 1;
        Expr b = 2;
        Expr c = 3;
        Expr d = 4;
        var pre = a - (b + c) + (c * d);
        CompilerServices.InferenceType(pre);

        var visitor = new DataFlowRewriter(new SimpleRule(), new());
        var post = visitor.Rewrite(pre);
        Assert.True(visitor.IsMutated);

        // rewrite once replace all:
        // if (post is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Sub } } root_call)
        // {
        //     if (root_call[IR.Math.Binary.Lhs] is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Sub } } lhs_call)
        //     {
        //         Assert.True(lhs_call[IR.Math.Binary.Rhs] is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Sub } });
        //     }
        // }

        // rewrite once replace once:
        if (post is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Sub } } root_call)
        {
            if (root_call[IR.Math.Binary.Lhs] is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Sub } } lhs_call)
            {
                Assert.True(lhs_call[IR.Math.Binary.Rhs] is Call { Target: IR.Math.Binary { BinaryOp: BinaryOp.Add } });
            }
        }
    }

    [Fact]
    public void TestMatchUpdatedVargs()
    {
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 2, 3, 4 });
        var lhs_unary = Unary(UnaryOp.Sin, input);
        var rhs_unary = Unary(UnaryOp.Cos, input);
        var pre = Binary(BinaryOp.Add, lhs_unary, rhs_unary);

        CompilerServices.InferenceType(pre);

        // update the lhs_unary
        var updated_lhs_unary = Unary(UnaryOp.Tanh, input);
        CompilerServices.InferenceType(updated_lhs_unary);

        // start match
        var pattern = IsCall("root", IsWildcard(), IsVArgs(
            "root_inputs",
            new Pattern[]
            {
                IsUnary(null, "lhs", _ => true, IsWildcard()),
                IsUnary(null, "rhs", _ => true, IsWildcard()),
            }));

        lhs_unary.ReplaceAllUsesWith(updated_lhs_unary);
        Assert.True(CompilerServices.TryMatchRoot(pre, pattern, new(), out var result));
        var root_inputs = (IReadOnlyList<Expr>)result!["root_inputs"];
        var lhs = (Call)result["lhs"];
        Assert.True(object.ReferenceEquals(root_inputs[0], lhs));
        Assert.True(lhs is Call { Target: Unary { UnaryOp: UnaryOp.Tanh } });
    }

    private sealed class SimpleRule : IRewriteRule
    {
        public IPattern Pattern { get; } = IsBinary(BinaryOp.Add, IsWildcard("lhs"), IsWildcard("rhs"));

        public Expr? GetReplace(IMatchResult result, RunPassContext options)
        {
            return (Expr)result["lhs"] - (Expr)result["rhs"];
        }
    }
}
