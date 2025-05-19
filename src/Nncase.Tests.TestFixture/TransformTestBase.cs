// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Nncase;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Analysis;
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

    public virtual BaseExpr TestMatched<T>(Function pre, IReadOnlyDictionary<IVar, IValue>? feeds = null)
        where T : IRewriteRule, new()
    {
        return TestMatchedCore(pre, feeds, false, new T());
    }

    public virtual BaseExpr TestMatched<T>(BaseExpr pre, IReadOnlyDictionary<IVar, IValue>? feeds = null)
        where T : IRewriteRule, new()
    {
        return TestMatchedCore(pre, feeds, new T());
    }

    public void CondMatch<T>(bool cond, BaseExpr expr)
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

    public Expr TestMatchedCore(Function pre, IReadOnlyDictionary<IVar, IValue>? feeds = null, bool isNotMatch = false, params IRewriteRule[] rules)
    {
        IAnalyzerManager analyzerManager = CompileSession.GetRequiredService<IAnalyzerManager>();
        var analysis = new Dictionary<System.Type, IAnalysisResult> { [typeof(IExprUserAnalysisResult)] = analyzerManager.GetAnaylsis<IExprUserAnalysisResult>(pre) };
        Assert.True(CompilerServices.InferenceType(pre), "TestInferFailed:" + pre.CheckedType);

        if (rules.Length == 0)
        {
            throw new InvalidOperationException("Rules should not be empty");
        }

        var preHashCode = pre.GetHashCode();
        var post = (Function)CompilerServices.Rewrite(pre, rules, new() { AnalysisResults = analysis, Driver = new DataflowPass() });
        if (isNotMatch)
        {
            Assert.Equal(preHashCode, post.GetHashCode());
        }
        else
        {
            var v1 = CompilerServices.Evaluate(pre.Body, feeds);
            Assert.NotEqual(preHashCode, post.GetHashCode());
            var v2 = CompilerServices.Evaluate(post.Body, feeds);
            if (!Comparator.AllEqual(v1, v2))
            {
                Comparator.Compare(v1, v2);
            }
        }

        return post;
    }

    public BaseExpr TestMatchedCore(BaseExpr pre, IReadOnlyDictionary<IVar, IValue>? feeds = null, params IRewriteRule[] rules)
    {
        pre.InferenceType();
        Assert.True(pre.InferenceType(), "TestInferFailed:" + pre.CheckedType);
        if (rules.Length == 0)
        {
            throw new InvalidOperationException("Rules should not be empty");
        }

        var preHashCode = pre.GetHashCode();
        var v1 = pre.Evaluate(feeds);
        var post = CompilerServices.Rewrite(pre, rules, new() { Driver = new DataflowPass() });
        Assert.NotEqual(preHashCode, post.GetHashCode());
        var v2 = post.Evaluate(feeds);
        if (!Comparator.AllEqual(v1, v2))
        {
            Comparator.Compare(v1, v2);
        }

        return post;
    }

    public void TestNotMatch(BaseExpr pre, params IRewriteRule[] rules)
    {
        pre.InferenceType();
        var preHashCode = pre.GetHashCode();
        var post = CompilerServices.Rewrite(pre, rules, new() { Driver = new DataflowPass() });
        Assert.Equal(preHashCode, post.GetHashCode());
    }

    public void TestNotMatch<T>(Function pre)
        where T : IRewriteRule, new()
    {
        TestMatchedCore(pre, null, true, new T());
    }

    public void TestNotMatch<T>(BaseExpr pre)
        where T : IRewriteRule, new()
    {
        TestNotMatch(pre, new T());
    }

    // public void TestSwappableBinary<T>(BinaryOp op, Expr lhs, Expr rhs) where T : IRewriteRule, new()
    // {
    //     TestMatched<T>(Binary(op, lhs, rhs));
    //     TestMatched<T>(Binary(op, rhs, lhs));
    // }
    public BaseExpr Rewrite<T>(BaseExpr pre, RunPassContext passOptions)
        where T : IRewriteRule, new()
    {
        return CompilerServices.Rewrite(pre, new IRewriteRule[] { new T() }, passOptions);
    }

    public BaseExpr RewriteWithSeq(BaseExpr expr, RunPassContext passOptions, IEnumerable<IRewriteRule> rules) =>
        rules.Aggregate(expr, (expr1, rule) => CompilerServices.Rewrite(expr1, new[] { rule }, passOptions));

    public BaseExpr RewriteWithSeq(BaseExpr expr, RunPassContext passOptions, IEnumerable<IRewriteRule> lower, IEnumerable<IRewriteRule> folds, IEnumerable<IRewriteRule> fuse)
    {
        var l = RewriteWithSeq(expr, passOptions, lower);
        var s = RewriteWithSeq(l, passOptions with { RewriteOnce = false }, folds);
        var f = RewriteWithSeq(s, passOptions, fuse);
        return f;
    }

    public BaseExpr FoldNop(BaseExpr expr, RunPassContext passOptions) =>
        CompilerServices.Rewrite(
            expr,
            new IRewriteRule[] { new FoldNopCast(), new FoldNopReshape(), },
            passOptions with { RewriteOnce = false });

    public BaseExpr RewriteWithSeq(BaseExpr expr, RunPassContext passOptions, IEnumerable<IRewriteRule> lower, IRewriteRule fold, IEnumerable<IRewriteRule> fuse) => RewriteWithSeq(expr, passOptions, lower, new[] { fold }, fuse);

    public BaseExpr RewriteWithSeq(BaseExpr expr, RunPassContext passOptions, IEnumerable<IRewriteRule> lower, IEnumerable<IRewriteRule> fuse) =>
        RewriteWithSeq(
            expr,
            passOptions,
            lower,
            new IRewriteRule[] { new FoldNopReshape(), new FoldNopCast(), },
            fuse);

    public BaseExpr TestMultiMatched<T>(BaseExpr expr, int count)
        where T : IRewriteRule, new()
        =>
            Enumerable.Range(0, count).Aggregate(expr, (expr1, i) =>
            {
                var ex = TestMatched<T>(expr1);
                return ex;
            });

    public BaseExpr RewriteMultiTimes<T>(BaseExpr expr, int count)
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
