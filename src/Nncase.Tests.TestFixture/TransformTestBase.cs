// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Quantization;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests;

// impl mixin by inherit interface with method had been impl
public partial class TransformTestBase : TestClassBase
{
    public TransformTestBase()
    {
        CompileOptions.QuantizeOptions.QuantType = DataTypes.UInt8;
        CompileOptions.QuantizeOptions.WQuantType = DataTypes.UInt8;
    }

    public virtual Expr TestMatched<T>(Expr pre)
        where T : IRewriteRule, new()
    {
        return TestMatchedCore(pre, new T());
    }

    public void CondMatch<T>(bool cond, Expr expr)
        where T : IRewriteRule, new()
    {
        if (cond)
        {
            TestMatched<T>(expr);
        }
        else
        {
            TestNotMatch<T>(expr);
        }
    }

    public Expr TestMatchedCore(Expr pre, params IRewriteRule[] rules)
    {
        Assert.True(pre.InferenceType(), "TestInferFailed:" + pre.CheckedType);
        if (rules.Length == 0)
        {
            throw new InvalidOperationException("Rules should not be empty");
        }

        var post = CompilerServices.Rewrite(pre, rules, new());
        Assert.NotEqual(pre, post);
        var v1 = pre.Evaluate();
        var v2 = post.Evaluate();

        Comparator.Compare(v1, v2);
        return post;
    }

    public void TestNotMatch(Expr pre, params IRewriteRule[] rules)
    {
        pre.InferenceType();
        var post = CompilerServices.Rewrite(pre, rules, new());
        Assert.Equal(pre, post);
    }

    public void TestNotMatch<T>(Expr pre)
        where T : IRewriteRule, new()
    {
        TestNotMatch(pre, new T());
    }

    // public void TestSwappableBinary<T>(BinaryOp op, Expr lhs, Expr rhs) where T : IRewriteRule, new()
    // {
    //     TestMatched<T>(Binary(op, lhs, rhs));
    //     TestMatched<T>(Binary(op, rhs, lhs));
    // }
    public Expr Rewrite<T>(Expr pre, RunPassContext passOptions)
        where T : IRewriteRule, new()
    {
        return CompilerServices.Rewrite(pre, new IRewriteRule[] { new T() }, passOptions);
    }

    public Expr RewriteWithSeq(Expr expr, RunPassContext passOptions, IEnumerable<IRewriteRule> rules) =>
        rules.Aggregate(expr, (expr1, rule) => CompilerServices.Rewrite(expr1, new[] { rule }, passOptions));

    public Expr RewriteWithSeq(Expr expr, RunPassContext passOptions, IEnumerable<IRewriteRule> lower, IEnumerable<IRewriteRule> folds, IEnumerable<IRewriteRule> fuse)
    {
        var l = RewriteWithSeq(expr, passOptions, lower);
        var s = RewriteWithSeq(l, passOptions with { RewriteOnce = false }, folds);
        var f = RewriteWithSeq(s, passOptions, fuse);
        return f;
    }

    public Expr FoldNop(Expr expr, RunPassContext passOptions) =>
        CompilerServices.Rewrite(
            expr,
            new IRewriteRule[]
            {
                new FoldNopCast(),
                new FoldNopReshape(),
            },
            passOptions with { RewriteOnce = false });

    public Expr RewriteWithSeq(Expr expr, RunPassContext passOptions, IEnumerable<IRewriteRule> lower, IRewriteRule fold, IEnumerable<IRewriteRule> fuse) => RewriteWithSeq(expr, passOptions, lower, new[] { fold }, fuse);

    public Expr RewriteWithSeq(Expr expr, RunPassContext passOptions, IEnumerable<IRewriteRule> lower, IEnumerable<IRewriteRule> fuse) =>
        RewriteWithSeq(
            expr,
            passOptions,
            lower,
            new IRewriteRule[]
            {
                new FoldNopReshape(),
                new FoldNopCast(),
            },
            fuse);

    public Expr TestMultiMatched<T>(Expr expr, int count)
        where T : IRewriteRule, new()
        =>
        Enumerable.Range(0, count).Aggregate(expr, (expr1, i) =>
        {
            var ex = TestMatched<T>(expr1);
            return ex;
        });

    public Expr RewriteMultiTimes<T>(Expr expr, int count)
        where T : IRewriteRule, new()
        =>
        Enumerable.Range(0, count).Aggregate(expr, (expr1, i) =>
        {
            var ex = Rewrite<T>(expr1, new());
            return ex;
        });

    protected virtual Task<TOutput> RunPassAsync<TInput, TOutput>(Pass<TInput, TOutput> pass, TInput input, bool rewriteOnce = true)
        where TInput : class
        where TOutput : class
    {
        var context = new RunPassContext { RewriteOnce = rewriteOnce };
        return pass.RunAsync(input, context);
    }
}
