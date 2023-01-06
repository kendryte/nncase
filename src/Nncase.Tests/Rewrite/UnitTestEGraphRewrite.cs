// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWriteTest;

public class UnitTestEGraphRewrite : TestClassBase
{
    [Fact]
    public void RewriteNoSenceAdd()
    {
        Var x = "a";
        var lhs = x + (100 / 120.0f) - 100;
        var y = lhs + 0;

        var pattern = PatternMatch.F.Math.Add(IsWildcard("lhs"), IsConst(0));

        var egraph = new EGraph();
        var root = egraph.Add(y);

        Assert.True(CompilerServices.TryMatchRoot(root.Nodes, pattern, out var eResults));
        Assert.Single(eResults);
        var wcxv = (Expr)eResults[0][pattern.Parameters[0]];
        Assert.Equal(wcxv, lhs);
        var to_eid = egraph.Add(wcxv);
        /*
          lhs + 0 <=> lhs
         */
        egraph.Union(to_eid, root);
        egraph.Rebuild();
    }

    [Fact]
    public void TestReassociate()
    {
        Expr pre = (Const)10 * 11 * 12;
        var rule = new Transform.Rules.Neutral.ReassociateMul();
        CompilerServices.ERewrite(pre, new[] { rule }, new());

        // Assert.Equal(newExpr, 10 * ((Const)11 * 12));
    }

    [Fact]
    public void TestClassicDemo()
    {
        var g = new EGraph();
        Var x = "x";
        var e1 = g.Add(x * 2);
        _ = g.Add(x * 2 / 2);
        var e2 = g.Add(x << 1);
        g.Union(e2, e1);
        g.Rebuild();
    }

    [Fact]
    public void TestTransposeBinaryMotion()
    {
        var c0 = (Call)NHWCToNCHW(Tensor.FromScalar(1, new[] { 2, 2, 3, 4 }));
        var c1 = (Call)NHWCToNCHW(Tensor.FromScalar(1, new[] { 2, 2, 1, 1 }));
        Assert.Equal(c0.Parameters[1].GetHashCode(), c1.Parameters[1].GetHashCode());

        Expr pre = c0 + c1;

        Assert.True(pre.InferenceType());

        var post = CompilerServices.ERewrite(pre, new[] { new Transform.Rules.Neutral.CombineTransposeBinary() }, new());

        Assert.True(post.InferenceType());
        Assert.Equal(pre.Evaluate(), post.Evaluate());
    }
}
